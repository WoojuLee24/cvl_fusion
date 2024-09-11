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
from pixloc.pixlib.models.mlp_mixer import MLPMixer
from pixloc.pixlib.models.simplevit import SimpleViT, Transformer, CrossTransformer
from pixloc.pixlib.geometry.optimization import optimizer_step

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
        nnrefine='v1.0',
        net='mlp',  # mlp, mixer vit1
        feature_dim=None,
        input='res',    # deprecated
        pool='none',    # 'embed_aap2'
        norm='none',
        pose_from='rt', # 'aa'
        pose_loss=False,
        main_loss='reproj2',    # reproj
        coe_lat=1.,
        coe_lon=1.,
        coe_rot=1.,
        trans_range=1.,
        rot_range=1.,
        range=False, # 'none',   # 'r', 't', 'rt'
        cascade=False,
        linearp='basic', # 'none', 'basic', 'pointnet', 'pointnet2', 'pointnet2_msg'
        radius=0.2,
        version=1.0,
        attention=False,
        mask=True,
        weights=False,
        input_dim=[128, 128, 32],  # [32, 128, 128],
        normalize_geometry='none',
        normalize_geometry_feature='l2', #'none',
        opt_list=False,
        jacobian=False,
        integral=False,
        kp=1.,
        kd=1.,
        ki=1.,
        multi_pose=1,
        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.conf = conf
        num_lambda = 3 if self.conf.pose_from == 'rt' else 6
        self.dampingnet = DampingNet(conf.damping,num_lambda)
        if self.conf.nnrefine == 'v1.0':
            self.nnrefine = NNrefinev1_0(conf)
        elif self.conf.nnrefine == 'v1.1':
            self.nnrefine = NNrefinev1_1(conf)
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
        # shift_gt = data['data']['shift_gt']
        shift_range = data['data']['shift_range']

        J_scaling = None
        if self.conf.normalize_features:
            F_query = torch.nn.functional.normalize(F_query, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()
        # shiftxyr = torch.zeros_like(shift_range)
        T_opt_list = []

        mean = data['data']['mean']
        std = data['data']['std']
        if self.conf.normalize_geometry == 'zsn':
            p3D = (p3D - p3D.mean()) / (p3D.std() + 1e-6)
        elif self.conf.normalize_geometry == 'l2':
            p3D = torch.nn.functional.normalize(p3D, dim=-1)
        elif self.conf.normalize_geometry == 'zsn2':
            p3D = (p3D - mean) / (std + 1e-6)

        for i in range(self.conf.num_iters):
            res, valid, w_unc, p3D_ref, F_ref2D, J = self.cost_fn.residual_jacobian3(T, self.conf.pose_from, *args)

            if self.conf.normalize_geometry == 'zsn':
                p3D_ref = (p3D_ref - p3D_ref.mean()) / (p3D_ref.std() + 1e-6)
            elif self.conf.normalize_geometry == 'l2':
                p3D_ref = torch.nn.functional.normalize(p3D_ref, dim=-1)
            elif self.conf.normalize_geometry == 'zsn2':
                p3D_ref = (p3D_ref - mean) / (std + 1e-6)

            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            delta = self.nnrefine(res, F_query, F_ref2D, p3D, p3D_ref, J, w_unc, valid,
                                  scale, lambda_, failed=failed)

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
                delta = delta * shift_range
                # shiftxyr += delta

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

            if self.conf.opt_list == True:
                T_opt_list.append(T)

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        if self.conf.opt_list == True:
            return T_opt_list, failed # , shiftxyr
        else:
            return T, failed # , shiftxyr


class NNrefinev1_0(nn.Module):
    def __init__(self, args):
        super(NNrefinev1_0, self).__init__()
        self.args = args
        self.p3d_mean = torch.tensor([[[0.3182,  1.6504, 14.9031]]], dtype=torch.float32).cuda()
        self.p3d_std = torch.tensor([[[9.1397,  0.0000, 10.4613]]], dtype=torch.float32).cuda()

        self.cin = self.args.input_dim  # [128, 128, 32]
        # self.cout = self.cin[0] * 3
        pointc = self.cin[2]
        self.initialize_rsum()

        # positional embedding
        self.linearp = nn.Sequential(nn.Linear(3, 16),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(16, pointc),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(pointc, pointc))

        # # channel projection
        # if self.args.version in [1.0]:
        #     self.cin = [c * 3 + pointc * 3 for c in self.cin]    # self.cin = [c * 3 + 2 * pointc for c in self.cin]
        # elif self.args.version in [1.01]:
        #     self.cin = [c * 3 + pointc * 2 for c in self.cin]
        # elif self.args.version in [1.02]:
        #     self.cin = [c * 3 + pointc for c in self.cin]
        # elif self.args.version in [1.03]:
        #     self.cin = [c + pointc * 3 for c in self.cin]
        # elif self.args.version in [1.04]:
        #     self.cin = [c * 3 for c in self.cin]
        # elif self.args.version in [1.05]:
        #     self.cin = [c for c in self.cin]
        #
        # if self.args.integral:
        #     I_size = self.args.input_dim
        #     self.cin = [c+I_size[i] for i, c in enumerate(self.cin)]
        # if self.args.jacobian:
        #     J_size = self.args.input_dim
        #     self.cin = [c+J_size[i]*3 for i, c in enumerate(self.cin)]

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        # self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
        #                              nn.Linear(self.cin[0], self.cout))
        # self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
        #                              nn.Linear(self.cin[1], self.cout))
        # self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
        #                              nn.Linear(self.cin[2], self.cout))

        # if self.args.pool == 'none':
        if self.args.net == 'mlp':  # default
            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 16)
                                         )
            self.cout = self.cin[0] * 3 * 16

            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout, 128),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(128, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())
        elif self.args.net == 'pmlp':
            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 16)
                                         )
            self.cout = self.cin[0] * 16

            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout, 128),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(128, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())
        elif self.args.net == 'jmlp':
            self.cout = self.cin[0] * 3
            hidden_dim = 512
            self.jmlp = nn.Sequential(nn.ReLU(inplace=False),
                                      nn.Linear(self.cout, hidden_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(hidden_dim, self.cout))

        elif self.args.net == 'jmlp2':
            jin = self.cin[0] * 3
            hidden_dim = 512

            self.jmlp = nn.Sequential(nn.ReLU(inplace=False),
                                      nn.Linear(jin, hidden_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(hidden_dim, jin))

            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 16)
                                         )
            self.cout = self.cin[0] * 3 * 16

            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout, 128),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(128, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())


        elif self.args.net in ['mixer', 'mixer_c', 'mixer_s']:
            self.mlp_mixer = MLPMixer(self.args.net,
                                      self.cout,  # in_channels
                                      self.args.max_num_points3D,  # num_patches
                                      hidden_size=self.cout,  # num_channels
                                      hidden_s=512,
                                      hidden_c=256,
                                      drop_p=0, off_act=False)

            self.pooling = nn.Sequential(nn.LayerNorm(self.args.max_num_points3D),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.GELU(),
                                         nn.LayerNorm(256),
                                         nn.Linear(256, 64),
                                         nn.GELU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

            self.mapping = nn.Sequential(nn.LayerNorm(self.cout),
                                         nn.Linear(self.cout, 128),
                                         nn.GELU(),
                                         nn.LayerNorm(128),
                                         nn.Linear(128, 32),
                                         nn.GELU(),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())
        elif args.net in ['tf1']:
            cin = self.cin[0]
            jin = self.cin[0] * 3
            hidden_dim = jin * 2
            self.j_sa = Transformer(dim=jin,
                                    depth=1,
                                    heads=8,
                                    dim_head=hidden_dim // 8,
                                    mlp_dim=jin)

            self.j_ca = CrossTransformer(dim1=jin,
                                         dim2=cin,
                                         depth=1,
                                         heads=8,
                                         dim_head=hidden_dim // 8,
                                         mlp_dim=jin)

            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, 16)
                                         )
            self.cout = jin * 16

            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout, 128),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(128, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())


    def forward(self, res, query_feat, ref_feat, p3D_query, p3D_ref, J, w_unc, valid,
                scale, lambda_, integral=False, failed=None):

        B, N, C = query_feat.size()

        if self.args.mask:
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            res = res * valid
            query_feat = query_feat * valid
            ref_feat = ref_feat * valid
            p3D_query = p3D_query * valid
            p3D_ref = p3D_ref * valid
            J = J * valid.unsqueeze(dim=-1).detach()
            w_unc = w_unc * valid
        if self.args.weights:
            res = res * w_unc
            J = J * w_unc.unsqueeze(dim=-1)

        # p3D_query_feat = self.linearp(p3D_query.contiguous())
        # p3D_ref_feat = self.linearp(p3D_ref.contiguous())
        #
        # # normalization
        # if self.args.normalize_geometry_feature == 'l2':
        #     p3D_query_feat = torch.nn.functional.normalize(p3D_query_feat, dim=-1)
        #     p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

        self.r = res

        # if self.args.integral:
        #     self.r_sum[2-scale] += res
        #     r = torch.cat([r, self.args.ki * self.r_sum[2-scale]], dim=-1)

        # if J is not None:
        #     J = J.view(B, N, -1)
        #     r = torch.cat([r, self.args.kd * J], dim=-1)

        if self.args.net == 'gd':
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            y = - lambda_ * Jtr.sum(dim=(1, 2))
        elif self.args.net == 'lm':
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            Jtr = w_unc[..., None] * Jtr
            Jtr = Jtr.sum(dim=(1, 2))

            Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
            Hess = w_unc[..., None] * Hess
            Hess = Hess.sum(-3)  # ... x 6 x6

            y = optimizer_step(Jtr, Hess, lambda_, mask=~failed)

        elif self.args.net == 'mlp':
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            Jtr = Jtr.view(B, N, -1)
            x = Jtr.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = - self.mapping(x)
        elif self.args.net == 'pmlp':
            # Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            # Jtr = Jtr.view(B, N, -1)
            x = res.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = - self.mapping(x)
        elif self.args.net == 'jmlp':
            J = J.view(B, N, -1)
            J = self.jmlp(J) + J
            J = J.view(B, N, -1, 3)
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            y = - Jtr.sum(dim=(1, 2))
        elif self.args.net == 'jmlp2':
            J = J.view(B, N, -1)
            J = self.jmlp(J) + J
            J = J.view(B, N, -1, 3)
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            Jtr = Jtr.view(B, N, -1)
            x = Jtr.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = - self.mapping(x)
        elif self.args.net == 'tf1':
            J = J.view(B, N, -1)
            J = self.j_sa(J)  # J^t@J
            J = self.j_ca(J, res)
            J = J.view(B, N, -1, 3)
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            y = - Jtr.sum(dim=(1, 2))
        elif self.args.net == 'tf2':
            J = J.view(B, N, -1)
            J = self.j_sa(J)  # J^t@J
            J = self.j_ca(J, res)
            J = J.view(B, N, -1, 3)
            Jtr = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
            Jtr = Jtr.view(B, N, -1)
            x = Jtr.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = - self.mapping(x)

        # B, N, C = r.shape
        # if 2-scale == 0:
        #     x = self.linear0(r)
        # elif 2-scale == 1:
        #     x = self.linear1(r)
        # elif 2-scale == 2:
        #     x = self.linear2(r)
        #
        # if self.args.net in ['mlp', 'mlp1', 'mlp2', 'mlp2.1', 'mlp2.2']:
        #     x = x.contiguous().permute(0, 2, 1).contiguous()
        #     x = self.pooling(x)
        #     x = x.view(B, -1)
        #     y = self.mapping(x)  # [B, 3]
        # elif self.args.net in ['mixer', 'mixer_c', 'mixer_s']:
        #     x = self.mlp_mixer(x)
        #     x = x.contiguous().permute(0, 2, 1).contiguous()
        #     x = self.pooling(x)
        #     x = x.view(B, -1)
        #     y = self.mapping(x)  # [B, 3]

        return y

    def initialize_rsum(self):
        self.r_sum = {0: 0, 1: 0, 2:0}


class NNrefinev1_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev1_1, self).__init__()
        self.args = args
        self.p3d_mean = torch.tensor([[[0.3182,  1.6504, 14.9031]]], dtype=torch.float32).cuda()
        self.p3d_std = torch.tensor([[[9.1397,  0.0000, 10.4613]]], dtype=torch.float32).cuda()

        self.dim = 64

        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.args.input_dim[0], self.dim))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.args.input_dim[1], self.dim))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(self.args.input_dim[2], self.dim))

        self.linear = [self.linear0, self.linear1, self.linear2]

        if self.args.jacobian:
            J_size = [input_dim*3 for input_dim in self.args.input_dim]
            self.j_linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                           nn.Linear(J_size[0], self.dim*3))
            self.j_linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                           nn.Linear(J_size[1], self.dim*3))
            self.j_linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                           nn.Linear(J_size[2], self.dim*3))

        self.j_linear = [self.j_linear0, self.j_linear1, self.j_linear2]

        self.initialize_rsum()

        # positional embedding
        self.linearp = nn.Sequential(nn.Linear(3, 16),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(16, self.dim),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(self.dim, self.dim))

        # channel projection
        if self.args.version in [1.0]:
            self.cout = self.dim * 6
        elif self.args.version in [1.01]:
            self.cout= self.dim * 5
        elif self.args.version in [1.02]:
            self.cout = self.dim * 3
        elif self.args.version in [1.03]:
            self.cout = self.dim * 4
        elif self.args.version in [1.04]:
            self.cout = self.dim * 3
        elif self.args.version in [1.05]:
            self.cout = self.dim

        if self.args.integral:
            self.cout = self.cout + self.dim
        if self.args.jacobian:
            self.cout = self.cout + self.dim*3
            if self.args.net == 'tp2':
                self.cout = self.cout + self.dim*9

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3


        # if self.args.pool == 'none':
        if self.args.net in ['mlp', 'tp1', 'tp2']:
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

        elif self.args.net == 'mlp2':
            self.pooling = nn.Sequential(nn.LayerNorm(self.args.topk),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.GELU(),
                                         nn.LayerNorm(256),
                                         nn.Linear(256, 64),
                                         nn.GELU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

            self.mapping = nn.Sequential(nn.LayerNorm(self.cout),
                                         nn.Linear(self.cout, 128),
                                         nn.GELU(),
                                         nn.LayerNorm(128),
                                         nn.Linear(128, 32),
                                         nn.GELU(),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())

        elif self.args.net == 'mlp2.1':
            self.pooling = nn.Sequential(#nn.LayerNorm(self.args.topk),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.GELU(),
                                         #nn.LayerNorm(256),
                                         nn.Linear(256, 64),
                                         nn.GELU(),
                                         #nn.LayerNorm(64),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

            self.mapping = nn.Sequential(#nn.LayerNorm(self.cout),
                                         nn.Linear(self.cout, 128),
                                         nn.GELU(),
                                         #nn.LayerNorm(128),
                                         nn.Linear(128, 32),
                                         nn.GELU(),
                                         #nn.LayerNorm(32),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())

        elif self.args.net == 'mlp2.2':
            self.pooling = nn.Sequential(nn.GELU(),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 64),
                                         nn.GELU(),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

            self.mapping = nn.Sequential(nn.GELU(),
                                         nn.Linear(self.cout, 128),
                                         nn.GELU(),
                                         nn.Linear(128, 32),
                                         nn.GELU(),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())

        elif self.args.net in ['mixer', 'mixer_c', 'mixer_s']:
            self.mlp_mixer = MLPMixer(self.args.net,
                                      self.cout,  # in_channels
                                      self.args.max_num_points3D,  # num_patches
                                      hidden_size=self.cout,  # num_channels
                                      hidden_s=512,
                                      hidden_c=256,
                                      drop_p=0, off_act=False)

            self.pooling = nn.Sequential(nn.LayerNorm(self.args.max_num_points3D),
                                         nn.Linear(self.args.max_num_points3D, 256),
                                         nn.GELU(),
                                         nn.LayerNorm(256),
                                         nn.Linear(256, 64),
                                         nn.GELU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 16)
                                         )
            self.cout *= 16

            self.mapping = nn.Sequential(nn.LayerNorm(self.cout),
                                         nn.Linear(self.cout, 128),
                                         nn.GELU(),
                                         nn.LayerNorm(128),
                                         nn.Linear(128, 32),
                                         nn.GELU(),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())

        elif self.args.net in ['tf1']:
            self.j_sa = Transformer(dim=self.dim*3,
                                    depth=1,
                                    heads=8,
                                    dim_head=self.dim // 8,
                                    mlp_dim=self.dim * 3)


            self.j_ca = CrossTransformer(dim1=self.dim*3,
                                         dim2=self.dim,
                                         depth=1,
                                         heads=8,
                                         dim_head=self.dim // 8,
                                         mlp_dim=self.dim * 3)

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



    def forward(self, query_feat, ref_feat, p3D_query, p3D_ref, scale, J=None, integral=False):

        B, N, C = query_feat.size()

        # query_feat = self.linear[2-scale](query_feat)
        # ref_feat = self.linear[2-scale](ref_feat) # edited

        p3D_query_feat = self.linearp(p3D_query.contiguous())
        p3D_ref_feat = self.linearp(p3D_ref.contiguous())

        # normalization
        if self.args.normalize_geometry_feature == 'l2':
            # query_feat = torch.nn.functional.normalize(query_feat, dim=-1)
            # ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
            p3D_query_feat = torch.nn.functional.normalize(p3D_query_feat, dim=-1)
            p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

        res = query_feat - ref_feat

        if self.args.version in [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]:    # resconcat2
            if self.args.version == 1.0:
                r = torch.cat([query_feat, ref_feat, self.args.kp * res,
                               p3D_query_feat, p3D_ref_feat, p3D_query_feat - p3D_ref_feat], dim=-1)
            elif self.args.version == 1.01:
                r = torch.cat([query_feat, ref_feat, self.args.kp * res,
                               p3D_query_feat, p3D_ref_feat], dim=-1)
            elif self.args.version == 1.02:
                r = torch.cat([query_feat, ref_feat, self.args.kp * res,
                               p3D_ref_feat], dim=-1)
            elif self.args.version == 1.03:
                r = torch.cat([self.args.kp * res,
                               p3D_query_feat, p3D_ref_feat, p3D_query_feat - p3D_ref_feat], dim=-1)
            elif self.args.version == 1.04:
                r = torch.cat([query_feat, ref_feat, self.args.kp * res], dim=-1)
            elif self.args.version == 1.05:
                r = torch.cat([self.args.kp * res], dim=-1)

        if self.args.integral:
            self.r_sum[2 - scale] += res
            r = torch.cat([r, self.args.ki * self.r_sum[2-scale]], dim=-1)

        if self.args.jacobian:
            # J = J.view(B, N, -1)
            # J = self.j_linear[2 - scale](J)
            # J = self.j_sa(J) if self.args.net in ['tf1'] else J # J^t@J
            # J = self.j_ca(J, res) if self.args.net in ['tf1'] else J
            # r = torch.cat([r, self.args.kd * J], dim=-1)

            if self.args.net == 'tf1':
                J = J.view(B, N, -1)
                J = self.j_sa(J)  # J^t@J
                J = self.j_ca(J, res)
            elif self.args.net == 'tp1':
                J = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
                J = J.reshape(B, N, -1).contiguous()
            elif self.args.net == 'tp2':
                Hess = torch.einsum('...ni,...nj->...nij', J, J)
                Hess = Hess.reshape(B, N, -1).contiguous()
                J = torch.einsum('...di,...dk->...di', J, res.unsqueeze(dim=-1))
                J = J.reshape(B, N, -1).contiguous()
                J = torch.cat([J, Hess], dim=-1)
            else:
                J = J.reshape(B, N, -1).contiguous()

            r = torch.cat([r, self.args.kd * J], dim=-1)

        if self.args.net in ['mlp', 'mlp2', 'mlp2.1', 'mlp2.2', 'tp1', 'tp2']:
            x = r.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif self.args.net in ['mixer', 'mixer_c', 'mixer_s']:
            x = self.mlp_mixer(r)
            x = x.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif self.args.net in ['tf1']:
            x = r.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]

        return y

    def initialize_rsum(self):
        self.r_sum = {0: 0, 1: 0, 2:0}