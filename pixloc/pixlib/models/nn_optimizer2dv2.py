import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa
from pixloc.pixlib.geometry.wrappers import project_grd_to_map

from .pointnet import PointNetEncoder
from .pointnet2 import PointNetEncoder2

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


class NNOptimizer2D(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        feature_dim=None,
        input='res',
        pool='aap2',
        norm='none',
        pose_from='rt', # 'aa',
        pose_loss=False,
        main_loss='reproj2',
        coe_lat=1.,
        coe_lon=1.,
        coe_rot=1.,
        trans_range=1.,
        rot_range=1.,
        range=False,  # 'none',   # 'r', 't', 'rt'
        cascade=False,
        linearp='basic', # 'none', 'basic', 'pointnet', 'pointnet2', 'pointnet2_msg'
        version=1.0,
        attention=False,
        mask=True, # False,
        input_dim=[128, 128, 32],  # [32, 128, 128],
        pool_rgb='none',
        # deprecated entries
        lambda_=0.,
        learned_damping=True,
        jacobian=True,
        kp=1.,
        kd=1.,
        ki=1.,
    )

    def _init(self, conf):
        self.conf = conf
        self.dampingnet = DampingNet(conf.damping)
        if conf.version == 0.1:
            self.nnrefine_rgb = NNrefinev0_1(conf)  # g2sp
        elif conf.version == 0.2:
            self.nnrefine_rgb = NNrefinev0_2(conf)  # s2gp
        elif conf.version == 0.3:
            self.nnrefine_rgb = NNrefinev0_3(conf)  # bp
        self.uv_pred = None
        self.uv_gt = None
        assert conf.learned_damping
        super()._init(conf)


    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['F_q2r'], data['F_r2q'],
            data['T_init'], data['cam_ref'], data['cam_q'],
            data['mask_q2r'], data['mask_r2q'], data.get('W_ref_q'), data['data'],
            data['scale'], data['version'])


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor, F_q2r: Tensor, F_r2q: Tensor,
             T_init: Pose, cam_ref: Camera, cam_q: Camera,
             mask_q2r: Optional[Tensor] = None, mask_r2q: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor, int]] = None,
             data=None,
             scale=None,
             version=1.0):

        T = T_init
        # shift_gt = data['data']['shift_gt']
        shift_range = data['shift_range']

        J_scaling = None
        # if self.conf.normalize_features:
        #     F_q_key = torch.nn.functional.normalize(F_q_key, dim=-1)
        args = (cam_ref, p3D, F_ref, F_q2r, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        # lambda_ = self.dampingnet()
        shiftxyr = torch.zeros_like(shift_range)

        for i in range(self.conf.num_iters):
            # if self.conf.jacobian:
            #     res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian2(T, *args)
            # else:
            #     valid, F_ref2D = self.cost_fn.residuals2(T, *args)
            #     J = None

            # # solve the nn optimizer

            if self.conf.version == 0.1:
                input_features = (F_q2r, F_ref, scale)
            elif self.conf.version == 0.2:
                input_features = (F_r2q, F_query, scale)
            elif self.conf.version == 0.3:
                input_features = (F_q2r, F_ref, F_r2q, F_query, scale)

            # delta = self.nnrefine_rgb(F_q2r, F_ref, scale)
            delta = self.nnrefine_rgb(*input_features)

            # rescaling
            # delta = delta * shift_range
            # shiftxyr += delta
            #
            # dt, dw = delta.split([2, 1], dim=-1)
            # B = dw.size(0)
            #
            # cos = torch.cos(dw)
            # sin = torch.sin(dw)
            # zeros = torch.zeros_like(cos)
            # ones = torch.ones_like(cos)
            # dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
            # dR = dR.view(B, 3, 3)  # shape = [B,3,3]
            # dt = torch.cat([dt, zeros], dim=-1)
            #
            # T_delta = Pose.from_Rt(dR, dt)
            T_delta, shiftxyr = self.delta2Tdelta(delta, shift_range, shiftxyr)

            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, shiftxyr


    def delta2Tdelta(self, delta, shift_range, shiftxyr):
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

        return T_delta, shiftxyr


# class NNrefinev0_1(nn.Module):
#     def __init__(self, args):
#         super(NNrefinev0_1, self).__init__()
#         self.args = args
#
#         self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
#         self.cout = 128
#         pointc = 128
#
#         if self.args.linearp != 'none':
#             self.cin = [c+pointc for c in self.cin]
#             if self.args.linearp == 'basic':
#                 self.linearp = nn.Sequential(nn.Linear(3, 16),
#                                              # nn.BatchNorm1d(16),
#                                              nn.ReLU(inplace=False),
#                                              nn.Linear(16, pointc),
#                                              # nn.BatchNorm1d(pointc),
#                                              nn.ReLU(inplace=False),
#                                              nn.Linear(pointc, pointc))
#             elif self.args.linearp == 'pointnet':
#                 self.linearp = nn.Sequential(PointNetEncoder(), # (B, N, 1088)
#                                              nn.ReLU(inplace=False),
#                                              nn.Linear(1088, pointc)
#                                              )
#             elif self.args.linearp in ['pointnet2', 'pointnet2_msg']:
#                 if self.args.linearp == 'pointnet2':
#                     linearp_property = [0.2, 32, [64,64,128]] # radius, nsample, mlp
#                     output_dim = linearp_property[2][-1]
#                 elif self.args.linearp == 'pointnet2_msg':
#                     linearp_property = [[0.1, 0.2, 0.4], [16, 32, 128], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]] # radius_list, nsample_list, mlp_list
#                     output_dim = torch.sum(torch.tensor(linearp_property[2], requires_grad=False), dim=0)[-1]
#                 self.linearp = nn.Sequential(PointNetEncoder2(self.args.max_num_points3D, linearp_property[0], linearp_property[1], linearp_property[2], self.args.linearp), # (B, N, output_dim)
#                                              nn.ReLU(inplace=False),
#                                              nn.Linear(output_dim, pointc)
#                                              )
#         else:
#             self.cin = [c+3 for c in self.cin]
#
#
#         # channel projection
#         if self.args.input in ['concat']:
#             self.cin = [c*2 for c in self.cin]
#
#         if self.args.pose_from == 'aa':
#             self.yout = 6
#         elif self.args.pose_from == 'rt':
#             self.yout = 3
#
#         self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
#                                      nn.Linear(self.cin[0], self.cout))
#         self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
#                                      nn.Linear(self.cin[1], self.cout))
#         self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
#                                      nn.Linear(self.cin[2], self.cout))
#
#         # if self.args.pool == 'none':
#         if self.args.pool == 'embed':
#             self.pooling = nn.Sequential(nn.ReLU(inplace=False),
#                                          nn.Linear(self.args.max_num_points3D, 256),
#                                          nn.ReLU(inplace=False),
#                                          nn.Linear(256, 64),
#                                          nn.ReLU(inplace=False),
#                                          nn.Linear(64, 1)
#                                          )
#
#         elif self.args.pool == 'embed_aap2':
#             self.pooling = nn.Sequential(nn.ReLU(inplace=False),
#                                          nn.Linear(self.args.max_num_points3D, 256),
#                                          nn.ReLU(inplace=False),
#                                          nn.Linear(256, 64),
#                                          nn.ReLU(inplace=False),
#                                          nn.Linear(64, 16)
#                                          )
#             self.cout *= 16
#
#         self.mapping = nn.Sequential(nn.ReLU(inplace=False),
#                                      nn.Linear(self.cout, 128),
#                                      nn.ReLU(inplace=False),
#                                      nn.Linear(128, 32),
#                                      nn.ReLU(inplace=False),
#                                      nn.Linear(32, self.yout),
#                                      nn.Tanh())
#
#     def forward(self, query_feat, ref_feat, p3D_query, p3D_ref, scale):
#
#         B, N, C = query_feat.size()
#
#         # normalization
#         if self.args.norm == 'zsn':
#             query_feat = (query_feat - query_feat.mean()) / (query_feat.std() + 1e-6)
#             ref_feat = (ref_feat - ref_feat.mean()) / (ref_feat.std() + 1e-6)
#         else:
#             pass
#
#         if self.args.linearp != 'none':
#             p3D_query = p3D_query.contiguous()
#             p3D_query_feat = self.linearp(p3D_query)
#             p3D_ref = p3D_ref.contiguous()
#             p3D_ref_feat = self.linearp(p3D_ref)
#         else:
#             p3D_query_feat = p3D_query.contiguous()
#             p3D_ref_feat = p3D_ref.contiguous()
#
#         query_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
#         ref_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)
#
#
#         if self.args.input == 'concat':     # default
#             r = torch.cat([query_feat, ref_feat], dim=-1)
#         else:
#             r = query_feat - ref_feat  # [B, C, H, W]
#
#         B, N, C = r.shape
#         if 2-scale == 0:
#             x = self.linear0(r)
#         elif 2-scale == 1:
#             x = self.linear1(r)
#         elif 2-scale == 2:
#             x = self.linear2(r)
#
#         if self.args.pool == 'max':
#             x = torch.max(x, 1, keepdim=True)[0]
#         elif 'embed' in self.args.pool:
#             x = x.contiguous().permute(0, 2, 1).contiguous()
#             x = self.pooling(x)
#         elif self.args.pool == 'avg':
#             x = torch.mean(x, 1, keepdim=True)
#
#         # if self.args.pool == 'none':
#         x = x.view(B, -1)
#         y = self.mapping(x)  # [B, 3]
#
#         return y


class NNrefinev0_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_1, self).__init__()
        self.args = args

        # channel projection
        self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
        self.cout = 32

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[0], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[1], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[2], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        if self.args.pool == 'gap':
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(64, 16),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(16, 3),
                                         nn.Tanh())

        elif self.args.pool_rgb == 'aap2':
            self.pool = nn.AdaptiveAvgPool2d((16, 16))
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(self.cout * 16 * 16, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, 3),
                                         nn.Tanh())

        elif self.args.pool_rgb == 'aap3':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(64 * 32 * 32, 2048),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(2048, 64),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(64, 3),
                                         nn.Tanh())


    def forward(self, F_q2r, F_ref, scale):

        B, C, _, _ = F_q2r.size()

        # normalization
        if self.args.norm == 'zsn':
            F_q2r = (F_q2r - F_q2r.mean()) / (F_q2r.std() + 1e-6)
            F_ref = (F_ref - F_ref.mean()) / (F_ref.std() + 1e-6)
        else:
            pass

        if self.args.input == 'concat':     # default
            r = torch.cat([F_q2r, F_ref], dim=-1)
        else:
            r = F_q2r - F_ref  # [B, C, H, W]

        B, C, _, _ = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2 - scale == 1:
            x = self.linear1(r)
        elif 2 - scale == 2:
            x = self.linear2(r)

        if self.args.pool == 'gap':
            x = torch.mean(x, dim=[2, 3])
            y = self.mapping(x)  # [B, 3]
        elif 'aap' in self.args.pool:
            x = self.pool(x)
            B, C, H, W = x.size()
            x = x.view(B, C*H*W)
            y = self.mapping(x)  # [B, 3]

        return y


class NNrefinev0_2(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_2, self).__init__()
        self.args = args

        # channel projection
        self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
        self.cout = 32

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[0], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[1], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[2], self.cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        if self.args.pool == 'gap':
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(64, 16),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(16, 3),
                                         nn.Tanh())

        elif self.args.pool_rgb == 'aap2':
            self.pool = nn.AdaptiveAvgPool2d((8, 32))
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(self.cout * 8 * 32, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, 3),
                                         nn.Tanh())

        elif self.args.pool_rgb == 'aap3':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(64 * 32 * 32, 2048),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(2048, 64),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(64, 3),
                                         nn.Tanh())


    def forward(self, F_r2q, F_q, scale):

        B, C, _, _ = F_r2q.size()

        if self.args.input == 'concat':     # default
            r = torch.cat([F_r2q, F_q], dim=-1)
        else:
            r = F_r2q - F_q  # [B, C, H, W]

        B, C, _, _ = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2 - scale == 1:
            x = self.linear1(r)
        elif 2 - scale == 2:
            x = self.linear2(r)

        if self.args.pool == 'gap':
            x = torch.mean(x, dim=[2, 3])
            y = self.mapping(x)  # [B, 3]
        elif 'aap' in self.args.pool:
            x = self.pool(x)
            B, C, H, W = x.size()
            x = x.view(B, C*H*W)
            y = self.mapping(x)  # [B, 3]

        return y



class NNrefinev0_3(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_3, self).__init__()
        self.args = args

        # channel projection
        self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
        self.cout = 32

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.q2r_linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[0], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))
        self.q2r_linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[1], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))
        self.q2r_linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[2], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))
        self.r2q_linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[0], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))
        self.r2q_linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[1], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))
        self.r2q_linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(self.cin[2], self.cout, kernel_size=(3, 3), stride=(1, 1),
                                                   padding=(1, 1)))

        if self.args.pool == 'gap':
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(64, 16),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(16, 3),
                                         nn.Tanh())

        elif self.args.pool_rgb == 'aap2':
            self.r2q_pool = nn.AdaptiveAvgPool2d((8, 32))
            self.r2q_mapping = nn.Sequential(nn.ReLU(inplace=False),
                                             nn.Linear(self.cout * 8 * 32, 1024),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(1024, 32))
            self.q2r_pool = nn.AdaptiveAvgPool2d((16, 16))
            self.q2r_mapping = nn.Sequential(nn.ReLU(inplace=False),
                                             nn.Linear(self.cout * 16 * 16, 1024),
                                             nn.ReLU(inplace=False),
                                             nn.Linear(1024, 32))
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(64, 16),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(16, 16),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(16, 3),
                                         nn.Tanh())


    def forward(self, F_q2r, F_ref, F_r2q, F_query, scale):

        B, C, A, A = F_q2r.size()
        B, C, H, W = F_r2q.size()

        r_q2r = F_q2r - F_ref  # [B, C, H, W]
        r_r2q = F_r2q - F_query

        if 2-scale == 0:
            r_q2r = self.q2r_linear0(r_q2r)
            r_r2q = self.r2q_linear0(r_r2q)
        elif 2 - scale == 1:
            r_q2r = self.q2r_linear1(r_q2r)
            r_r2q = self.r2q_linear1(r_r2q)
        elif 2 - scale == 2:
            r_q2r = self.q2r_linear2(r_q2r)
            r_r2q = self.r2q_linear2(r_r2q)

        if self.args.pool == 'gap':
            r_q2r = torch.mean(r_q2r, dim=[2, 3])
            q2r = self.q2r_mapping(r_q2r)  # [B, 3]
            r_r2q = torch.mean(r_r2q, dim=[2, 3])
            r2q = self.r2q_mapping(r_r2q)  # [B, 3]

            r2q_q2r = torch.cat([r2q, q2r], dim=1)
            y = self.mapping(r2q_q2r)

        elif 'aap' in self.args.pool:
            r_q2r = self.q2r_pool(r_q2r)
            b, c, h, w = r_q2r.size()
            r_q2r = r_q2r.view(b, c*h*w)
            q2r = self.q2r_mapping(r_q2r)  # [B, 3]

            r_r2q = self.r2q_pool(r_r2q)
            b, c, h, w = r_r2q.size()
            r_r2q = r_q2r.view(b, c*h*w)
            r2q = self.r2q_mapping(r_r2q)  # [B, 3]

            r2q_q2r = torch.cat([r2q, q2r], dim=1)
            y = self.mapping(r2q_q2r)

        return y

