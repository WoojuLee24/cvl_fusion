import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

from .pointnet import PointNetEncoder, PointNetEncoder1_1
from .pointnet2 import PointNetEncoder2
from .pointnet2_1 import PointNetEncoder2_1

logger = logging.getLogger(__name__)


class DampingNet(nn.Module):
    def __init__(self, conf, num_params=6):
        super().__init__()
        self.conf = conf
        if conf.type == 'constant':
            const = torch.zeros(num_params)
            self.register_parameter('const', torch.nn.Parameter(const))
        else:
            raise ValueError(f'Unsupported type of damping: {conf.type}.')

    def forward(self):
        min_, max_ = self.conf.log_range
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
        return lambda_


class NNOptimizer3D(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        feature_dim=None,
        input='res',
        pool='none',
        norm='none',
        pose_from='aa',
        pose_loss=False,
        main_loss='reproj',
        coe_lat=1.,
        coe_lon=1.,
        coe_rot=1.,
        trans_range=1.,
        rot_range=1.,
        range=False, # 'none',   # 'r', 't', 'rt'
        cascade=False,
        linearp='basic', # 'none', 'basic', 'pointnet', 'pointnet2', 'pointnet2_msg'
        radius=0.2,
        version=0.1,
        attention=False,
        mask=True,
        input_dim=[128, 128, 32],  # [32, 128, 128],
        normalize_geometry='none',
        normalize_geometry_feature='none',
        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.conf = conf
        self.dampingnet = DampingNet(conf.damping)
        self.nnrefine = NNrefinev0_1(conf)
        assert conf.learned_damping
        super()._init(conf)


    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['T_init'],
            data['camera'], data['mask'], data.get('W_ref_q'), data, data['scale'])


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor, int]] = None,
             data=None,
             scale=None):

        T = T_init
        shift_gt = data['data']['shift_gt']
        shift_range = data['data']['shift_range']

        J_scaling = None
        if self.conf.normalize_features:
            F_query = torch.nn.functional.normalize(F_query, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()
        shiftxyr = torch.zeros_like(shift_range)

        for i in range(self.conf.num_iters):
            # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
            valid, F_ref2D = self.cost_fn.residuals2(T, *args)

            if self.conf.normalize_geometry == 'zsn':
                p3D = (p3D - p3D.mean()) / (p3D.std() + 1e-6)
            elif self.conf.normalize_geometry == 'l2':
                p3D = torch.nn.functional.normalize(p3D, dim=-1)
            elif self.conf.normalize_geometry == 'zsn2':
                mean = torch.tensor([-0.1917,  0.9250, 15.6600]).to(p3D.device).repeat(1, 1, 1)
                std = torch.tensor([6.9589,  0.8642, 11.5166]).to(p3D.device).repeat(1, 1, 1)
                p3D = (p3D - mean) / (std + 1e-6)

            p3D_ref = T * p3D

            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            if self.conf.mask:
                valid = valid.float().unsqueeze(dim=-1).detach()
                F_query = F_query * valid
                F_ref2D = F_ref2D * valid
                p3D = p3D * valid
                p3D_ref = p3D_ref * valid

            delta = self.nnrefine(F_query, F_ref2D, p3D, p3D_ref, scale)

            if self.conf.pose_from == 'aa':
                # compute the pose update
                dt, dw = delta.split([3, 3], dim=-1)
                # dt, dw = delta.split([2, 1], dim=-1)
                # fix z trans, roll and pitch
                zeros = torch.zeros_like(dw[:,-1:])
                dw = torch.cat([zeros,zeros,dw[:,-1:]], dim=-1)
                dt = torch.cat([dt[:,0:2],zeros], dim=-1)
                T_delta = Pose.from_aa(dw, dt)
            elif self.conf.pose_from == 'rt':
                # rescaling
                mul_range = torch.tensor([[self.conf.trans_range, self.conf.trans_range, self.conf.rot_range]], dtype=torch.float32)
                mul_range = mul_range.to(shift_range.device)
                shift_range = shift_range * mul_range
                delta = delta * shift_range.detach()
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]

                dt = torch.cat([dt, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)

            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, shiftxyr


class NNrefinev0_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_1, self).__init__()
        self.args = args

        self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
        if self.args.version in [2.4, 2.5]:
            self.cin = [128, 128, 128]
        self.cout = 128
        pointc = 128

        if self.args.linearp != 'none':
            self.cin = [c+pointc for c in self.cin]
            if self.args.linearp == 'basic' or self.args.linearp == True:
                self.linearp = nn.Sequential(nn.Linear(3, 16),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(16, pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc))
            elif self.args.linearp == 'basicv2.3':
                self.linearp = nn.Sequential(nn.Linear(3, 16),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(16, pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc))
                self.linearp0 = nn.Linear(pointc, 32)
            elif self.args.linearp in ['basicv2.4', 'point2v2.4']:
                self.linearp = nn.Sequential(nn.Linear(3, 16),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(16, pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc))
                if self.args.linearp == 'basicv2.4':
                    self.linearp0 = nn.Sequential(nn.Linear(3, 16),
                                                  # nn.BatchNorm1d(16),
                                                  nn.ReLU(inplace=False),
                                                  nn.Linear(16, pointc),
                                                  # nn.BatchNorm1d(pointc),
                                                  nn.ReLU(inplace=False),
                                                  nn.Linear(pointc, pointc))
                elif self.args.linearp == 'point2v2.4':
                    linearp_property = [0.2, 32, [32, 32, 32]]  # radius, nsample, mlp
                    self.linearp0 = PointNetEncoder2_1(self.args.max_num_points3D,
                                                      linearp_property[0],
                                                      linearp_property[1],
                                                      linearp_property[2],
                                                      self.args.linearp) # (B, N, output_dim)

                self.linearp_r2 = nn.Sequential(nn.Linear(2 * pointc, 2 * pointc),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(2 * pointc, 2 * pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(2 * pointc, 2 * pointc))
                self.linear_rgb = nn.Linear(32, 128)
            elif self.args.linearp in ['basicv2.5', 'point2v2.5']:
                if self.args.linearp == 'basicv2.5':
                    self.linearp = nn.Sequential(nn.Linear(3, 16),
                                                 # nn.BatchNorm1d(16),
                                                 nn.ReLU(inplace=False),
                                                 nn.Linear(16, pointc),
                                                 # nn.BatchNorm1d(pointc),
                                                 nn.ReLU(inplace=False),
                                                 nn.Linear(pointc, pointc),
                                                 nn.ReLU(inplace=False),
                                                 nn.Linear(pointc, pointc)
                                                 )      # pointnet2.1?
                elif self.args.linearp == 'point2v2.5':
                    linearp_property = [0.2, 32, [32, 32, 32]]  # radius, nsample, mlp
                    self.linearp = PointNetEncoder2_1(self.args.max_num_points3D,
                                                      linearp_property[0],
                                                      linearp_property[1],
                                                      linearp_property[2],
                                                      self.args.linearp)  # (B, N, output_dim)


                self.linearp_geo = nn.Sequential(nn.Linear(pointc, pointc),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc))
                self.linear_rgb = nn.Linear(32, 128)

            elif self.args.linearp in ['basicv2.6', 'point2v2.6']:
                self.linearp_pe = nn.Sequential(nn.Linear(3, 16),
                                                # nn.BatchNorm1d(16),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(16, pointc),
                                                # nn.BatchNorm1d(pointc),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(pointc, pointc))

                self.linearp_geo2 = nn.Sequential(nn.Linear(pointc, pointc),
                                                  # nn.BatchNorm1d(16),
                                                  nn.ReLU(inplace=False),
                                                  nn.Linear(pointc, pointc),
                                                  # nn.BatchNorm1d(pointc),
                                                  nn.ReLU(inplace=False),
                                                  nn.Linear(pointc, pointc))

                if self.args.linearp == 'basicv2.6':
                    self.linearp_geo = nn.Sequential(nn.Linear(3, 16),
                                                     # nn.BatchNorm1d(16),
                                                     nn.ReLU(inplace=False),
                                                     nn.Linear(16, pointc),
                                                     # nn.BatchNorm1d(pointc),
                                                     nn.ReLU(inplace=False),
                                                     nn.Linear(pointc, pointc))

                elif self.args.linearp == 'point2v2.6':
                    linearp_property = [0.2, 32, [32, 32, 32]]  # radius, nsample, mlp
                    self.linearp_geo = PointNetEncoder2_1(self.args.max_num_points3D,
                                                          linearp_property[0],
                                                          linearp_property[1],
                                                          linearp_property[2],
                                                          self.args.linearp)  # (B, N, output_dim)

                self.linearp_r2 = nn.Sequential(nn.Linear(2 * pointc, 2 * pointc),
                                                # nn.BatchNorm1d(16),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(2 * pointc, 2 * pointc),
                                                # nn.BatchNorm1d(pointc),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(2 * pointc, 2 * pointc))

                self.linear_rgb = nn.Linear(32, 128)

            elif self.args.linearp == 'uv':
                self.linearp = nn.Sequential(nn.Linear(2, 16),
                                             # nn.BatchNorm1d(16),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(16, pointc),
                                             # nn.BatchNorm1d(pointc),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(pointc, pointc))
            elif self.args.linearp == 'pointnet':
                self.linearp = nn.Sequential(PointNetEncoder(), # (B, N, 1088)
                                             nn.ReLU(inplace=False),
                                             nn.Linear(1088, pointc)
                                             )
            elif self.args.linearp == 'pointnet1.1':
                self.linearp = nn.Sequential(PointNetEncoder1_1(),  # (B, N, 128)
                                             # nn.ReLU(inplace=False),
                                             # nn.Linear(1088, pointc)
                                             )
            elif self.args.linearp in ['pointnet2', 'pointnet2_msg']:
                if self.args.linearp == 'pointnet2':
                    linearp_property = [0.2, 32, [64,64,128]] # radius, nsample, mlp
                    output_dim = linearp_property[2][-1]
                elif self.args.linearp == 'pointnet2_msg':
                    linearp_property = [[0.1, 0.2, 0.4], [16, 32, 128], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]] # radius_list, nsample_list, mlp_list
                    output_dim = torch.sum(torch.tensor(linearp_property[2], requires_grad=False), dim=0)[-1]
                self.linearp = nn.Sequential(PointNetEncoder2(self.args.max_num_points3D, linearp_property[0], linearp_property[1], linearp_property[2], self.args.linearp), # (B, N, output_dim)
                                             nn.ReLU(inplace=False),
                                             nn.Linear(output_dim, pointc)
                                             )
            elif self.args.linearp in ['pointnet2.1', 'pointnet2.1_msg']:
                if self.args.linearp == 'pointnet2.1':
                    linearp_property = [self.args.radius, 32, [16, 16, 32]] # [0.2, 32, [64,64,128]] # radius, nsample, mlp
                    output_dim = linearp_property[2][-1]
                elif self.args.linearp == 'pointnet2.1_msg':
                    linearp_property = [[0.1, 0.2, 0.4], [16, 32, 128], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]] # radius_list, nsample_list, mlp_list
                    output_dim = torch.sum(torch.tensor(linearp_property[2], requires_grad=False), dim=0)[-1]
                self.linearp = PointNetEncoder2_1(self.args.max_num_points3D,
                                                  linearp_property[0], linearp_property[1], linearp_property[2], self.args.linearp) # (B, N, output_dim)

        else:
            self.cin = [c+3 for c in self.cin]


        # channel projection
        if self.args.version == 1.1:
            self.cin = [c+3 for c in self.cin]
        elif self.args.version == 2.3:
            self.cin = [128, 128, 32]
        elif self.args.version == 2.4:
            self.cin = [c*2 for c in self.cin]
        if self.args.input in ['concat']:
            self.cin = [c*2 for c in self.cin]


        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.cin[0], self.cout))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.cin[1], self.cout))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.cin[2], self.cout))

        # if self.args.pool == 'none':
        if self.args.pool == 'embed':
            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 1)
                                         )

        elif self.args.pool == 'embed_aap2':
            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

        self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                     nn.Linear(self.cout, 128),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(128, 32),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(32, self.yout),
                                     nn.Tanh())


    def forward(self, query_feat, ref_feat, p3D_query, p3D_ref, scale):

        B, N, C = query_feat.size()

        ## normalization
        #if self.args.norm == 'zsn':
        #    query_feat = (query_feat - query_feat.mean()) / (query_feat.std() + 1e-6)
        #    ref_feat = (ref_feat - ref_feat.mean()) / (ref_feat.std() + 1e-6)
        #else:
        #    pass

        if self.args.version == 1.0:
            if self.args.linearp != 'none':
                p3D_query = p3D_query.contiguous()
                p3D_query_feat = self.linearp(p3D_query)
                p3D_ref = p3D_ref.contiguous()
                p3D_ref_feat = self.linearp(p3D_ref)
            else:
                p3D_query_feat = p3D_query.contiguous()
                p3D_ref_feat = p3D_ref.contiguous()

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_query_feat = torch.nn.functional.normalize(p3D_query_feat, dim=-1)
                p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

            query_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
            ref_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)

            if self.args.input == 'concat':     # default
                r = torch.cat([query_feat, ref_feat], dim=-1)
            else:
                r = query_feat - ref_feat  # [B, C, H, W]

        elif self.args.version == 1.1:
            p3D_query = p3D_query.contiguous()
            p3D_query_feat = self.linearp(p3D_query)
            p3D_query_feat = torch.cat([p3D_query_feat, p3D_query], dim=-1)
            p3D_ref = p3D_ref.contiguous()
            p3D_ref_feat = self.linearp(p3D_ref)
            p3D_ref_feat = torch.cat([p3D_ref_feat, p3D_query], dim=-1)

            query_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
            ref_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)

            if self.args.input == 'concat':  # default
                r = torch.cat([query_feat, ref_feat], dim=-1)
            else:
                r = query_feat - ref_feat  # [B, C, H, W]

        elif self.args.version == 1.2:
            if self.args.linearp != 'none':
                p3D_query = p3D_query.contiguous()
                p3D_query_feat = self.linearp(p3D_query)
                p3D_ref = p3D_ref.contiguous()
                p3D_ref_feat = self.linearp(p3D_ref)
            else:
                p3D_query_feat = p3D_query.contiguous()
                p3D_ref_feat = p3D_ref.contiguous()

            query_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
            ref_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)
            # added
            r0 = ref_feat - p3D_ref_feat

            if self.args.input == 'concat':     # default
                r = torch.cat([query_feat, ref_feat], dim=-1)
            else:
                r = query_feat - ref_feat  # [B, C, H, W]

        elif self.args.version == 2.0:
            if self.args.linearp != 'none':
                p3D_query = p3D_query.contiguous()
                p3D_feat = self.linearp(p3D_query)
            else:
                p3D_feat = p3D_query.contiguous()

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_feat = torch.nn.functional.normalize(p3D_feat, dim=-1)

            r = query_feat - ref_feat
            r = torch.cat([r, p3D_feat], dim=-1)

        elif self.args.version == 2.1:
            if self.args.linearp != 'none':
                p3D_ref = p3D_ref.contiguous()
                p3D_ref_feat = self.linearp(p3D_ref)
            else:
                p3D_ref_feat = p3D_ref.contiguous()

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

            r = ref_feat
            r = torch.cat([r, p3D_ref_feat], dim=-1)

        elif self.args.version == 2.2:
            if self.args.linearp != 'none':
                p3D_ref = p3D_ref.contiguous()
                p3D_ref_feat = self.linearp(p3D_ref)
            else:
                p3D_ref_feat = p3D_ref.contiguous()

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

            r = ref_feat - query_feat
            r = torch.cat([r, p3D_ref_feat], dim=-1)

        elif self.args.version == 2.3:
            if self.args.linearp != 'none':
                p3D_ref = p3D_ref.contiguous()
                p3D_ref_feat = self.linearp(p3D_ref)
                if scale == 0:
                    p3D_ref_feat = self.linearp0(p3D_ref_feat)
            else:
                p3D_ref_feat = p3D_ref.contiguous()

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

            r = ref_feat - p3D_ref_feat

        elif self.args.version == 2.4:
            if scale == 0:
                query_feat = self.linear_rgb(query_feat)    # 32 -> 128
                ref_feat = self.linear_rgb(ref_feat)    # 32 -> 128
                query_feat = torch.nn.functional.normalize(query_feat, dim=-1)
                ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
            p3D_query_feat = self.linearp(p3D_query.contiguous())
            p3D_ref_feat = self.linearp(p3D_ref.contiguous())

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_query_feat = torch.nn.functional.normalize(p3D_query_feat, dim=-1)
                p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

            query_rgb_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
            ref_rgb_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)

            r1 = query_rgb_feat - ref_rgb_feat  # [B, N, 2C]

            p3D_ref = p3D_ref.contiguous()
            p3D_ref_feat0 = self.linearp0(p3D_ref)

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_ref_feat0 = torch.nn.functional.normalize(p3D_ref_feat0, dim=-1)

            # ref_feat = ref_feat # linear projection required for geometric ref feat??
            r2 = torch.cat([ref_feat, p3D_ref_feat0], dim=-1)   # [B, N, 2C]
            r2 = self.linearp_r2(r2)    # [B, N, 2C]

            r = torch.cat([r1, r2], dim=-1)     # [B, N, 4C]

        elif self.args.version == 2.5:
            if scale == 0:
                query_feat = self.linear_rgb(query_feat)    # 32 -> 128
                ref_feat = self.linear_rgb(ref_feat)    # 32 -> 128
                query_feat = torch.nn.functional.normalize(query_feat, dim=-1)
                ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)

            p3D_q2r = p3D_ref.contiguous()
            p3D_q2r_feat = self.linearp(p3D_q2r)    # [B, N, C]
            ref_geo_feat = self.linearp_geo(ref_feat)

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_q2r_feat = torch.nn.functional.normalize(p3D_q2r_feat, dim=-1)
                ref_geo_feat = torch.nn.functional.normalize(ref_geo_feat, dim=-1)

            query_q2r_feat = torch.cat([query_feat, p3D_q2r_feat], dim=-1)

            ref_feat = torch.cat([ref_feat, ref_geo_feat], dim=-1)

            r = query_q2r_feat - ref_feat

        elif self.args.version == 2.6:
            if scale == 0:
                query_feat = self.linear_rgb(query_feat)    # 32 -> 128
                ref_feat = self.linear_rgb(ref_feat)    # 32 -> 128
                query_feat = torch.nn.functional.normalize(query_feat, dim=-1)
                ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
            p3D_query_pe = self.linearp_pe(p3D_query.contiguous())   # positional encoding
            p3D_q2r_pe = self.linearp_pe(p3D_ref.contiguous())   # positional encoding

            # normalization
            if self.args.normalize_geometry_feature == 'l2':
                p3D_query_pe = torch.nn.functional.normalize(p3D_query_pe, dim=-1)
                p3D_q2r_pe = torch.nn.functional.normalize(p3D_q2r_pe, dim=-1)

            query_rgb_feat = torch.cat([query_feat, p3D_query_pe], dim=2)
            ref_rgb_feat = torch.cat([ref_feat, p3D_q2r_pe], dim=2)

            r1 = query_rgb_feat - ref_rgb_feat  # [B, N, 2C]

            p3D_q2r_geofeat = self.linearp_geo(p3D_ref.contiguous())  # geometric encoding
            ref_geofeat = self.linearp_geo2(ref_feat)  # geometric encoding

            # # normalization
            # if self.args.normalize_geometry_feature == 'l2':
            #     p3D_ref_feat0 = torch.nn.functional.normalize(p3D_ref_feat0, dim=-1)
            #     ref_geo_feat = torch.nn.functional.normalize(ref_geo_feat, dim=-1)

            # ref_feat = ref_feat # linear projection required for geometric ref feat??
            r2 = 1 - F.cosine_similarity(ref_geofeat, p3D_q2r_geofeat, dim=-1)
            r2 = self.linearp_r2(r2)    # [B, N, 2C]

            r = torch.cat([r1, r2], dim=-1)     # [B, N, 4C]


        B, N, C = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        if self.args.pool == 'max':
            x = torch.max(x, 1, keepdim=True)[0]
        elif 'embed' in self.args.pool:
            x = x.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
        elif self.args.pool == 'avg':
            x = torch.mean(x, 1, keepdim=True)

        # if self.args.pool == 'none':
        x = x.view(B, -1)
        y = self.mapping(x)  # [B, 3]

        return y
