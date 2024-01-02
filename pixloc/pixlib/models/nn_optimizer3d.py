import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

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
        range=True, # 'none',   # 'r', 't', 'rt'
        shift_range='default', # default: half_range, range_same, same w.r.t data.shift_range
        cascade=False,
        linearp=False,
        attention=False,
        mask=False,
        input_dim=[128, 128, 32],  # [32, 128, 128],
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
            res, valid, w_unc, F_ref2D, info = self.cost_fn.residuals(T, *args)

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

            # # compute the cost and aggregate the weights
            # cost = (res**2).sum(-1)
            # cost, w_loss, _ = self.loss_fn(cost)
            # weights = w_loss * valid.float()
            # if w_unc is not None:
            #     weights = weights*w_unc
            # if self.conf.jacobi_scaling:
            #     J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # # solve the linear system
            # g, H = self.build_system(J, res, weights)
            # delta = optimizer_step(g, H, lambda_, mask=~failed)
            # if self.conf.jacobi_scaling:
            #     delta = delta * J_scaling

            # # solve the nn optimizer


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
                if self.conf.shift_range == 'same':
                    delta = 2 * delta * shift_range - shift_range
                elif self.conf.shift_range == 'range_same':
                    delta = 2 * delta * shift_range
                else:
                    delta = delta * shift_range
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


            if self.conf.range == True:
                shift = (T_delta @ T) @ T_init.inv()
                B = dt.size(0)
                t = shift.t[:, :2]
                rand_t = torch.distributions.uniform.Uniform(-1, 1).sample([B, 2]).to(dt.device)
                rand_t.requires_grad = True
                if self.conf.shift_range == 'same':
                    a = - 2 * shift_range[0][0] - shift_range[0][0]
                    b = 2 * shift_range[0][0] - shift_range[0][0]
                elif self.conf.shift_range == 'range_same':
                    a = - 2 * shift_range[0][0]
                    b = 2 * shift_range[0][0]
                else:
                    a = -shift_range[0][0]
                    b = shift_range[0][0]
                t = torch.where((t > a) & (t < b), t, rand_t)
                # t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
                zero = torch.zeros([B, 1]).to(t.device)
                # zero = shift.t[:, 2:3]
                t = torch.cat([t, zero], dim=1)
                shift._data[..., -3:] = t
                T = shift @ T_init  # TODO
            else:
                T = T_delta @ T

            # if self.conf.range in 'rt':
            #     shift = (T_delta @ T) @ T_init.inv()
            #     B = dt.size(0)
            #     if 't' in self.conf.range:
            #         t = shift.t[:, :2]
            #         rand_t = torch.distributions.uniform.Uniform(-1, 1).sample([B, 2]).to(dt.device)
            #         rand_t.requires_grad = True
            #         t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
            #         zero = torch.zeros([B, 1]).to(t.device)
            #         # zero = shift.t[:, 2:3]
            #         t = torch.cat([t, zero], dim=1)
            #         shift._data[..., -3:] = t
            #     if 'r' in self.conf.range:
            #         r = shift.R
            #         rand_r = torch.distributions.uniform.Uniform(-0.01, 0.01).sample([B, 1]).to(dt.device)
            #         rand_r.requires_grad = True
            #         r = torch.where((r > -shift_range[0][-1]) & (r < shift_range[0][-1]), r, rand_r)
            #         cos = torch.cos(r)
            #         sin = torch.sin(r)
            #         zeros = torch.zeros_like(cos)
            #         ones = torch.ones_like(cos)
            #         r = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
            #         shift._data[..., :-3] = r
            #     T = shift @ T_init  # TODO
            # else:
            #     T = T_delta @ T

            # self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
            #          valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
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
        self.cout = 128
        pointc = 128

        if self.args.linearp:
            self.cin = [c+pointc for c in self.cin]
            self.linearp = nn.Sequential(nn.Linear(3, 16),
                                         # nn.BatchNorm1d(16),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(16, pointc),
                                         # nn.BatchNorm1d(pointc),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(pointc, pointc))
        else:
            self.cin = [c+3 for c in self.cin]


        # channel projection
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
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout, 128),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(128, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())

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

        elif self.args.pool == 'aap2':
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.args.max_num_points3D * self.cout, 256),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(256, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, self.yout),
                                         nn.Tanh()
                                         )



    def forward(self, query_feat, ref_feat, p3D_query, p3D_ref, scale):

        B, N, C = query_feat.size()

        # normalization
        if self.args.norm == 'zsn':
            query_feat = (query_feat - query_feat.mean()) / (query_feat.std() + 1e-6)
            ref_feat = (ref_feat - ref_feat.mean()) / (ref_feat.std() + 1e-6)
        else:
            pass

        if self.args.linearp:
            p3D_query = p3D_query.contiguous()
            p3D_query_feat = self.linearp(p3D_query)
            p3D_ref = p3D_ref.contiguous()
            p3D_ref_feat = self.linearp(p3D_ref)
        else:
            p3D_query_feat = p3D_query.contiguous()
            p3D_ref_feat = p3D_ref.contiguous()

        query_feat = torch.cat([query_feat, p3D_query_feat], dim=2)
        ref_feat = torch.cat([ref_feat, p3D_ref_feat], dim=2)


        if self.args.input == 'concat':     # default
            r = torch.cat([query_feat, ref_feat], dim=-1)
        else:
            r = query_feat - ref_feat  # [B, C, H, W]

        B, N, C = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        if self.args.pool == 'max':
            x = torch.max(x, 1, keepdim=True)[0]
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif 'embed' in self.args.pool:
            x = x.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif self.args.pool == 'avg':
            x = torch.mean(x, 1, keepdim=True)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif self.args.pool == 'aap2':
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]

        return y
