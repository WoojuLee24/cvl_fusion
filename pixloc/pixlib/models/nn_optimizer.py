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


class NNOptimizer(BaseOptimizer):
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
        range=False,
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
            res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)

            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            if w_unc is not None:
                weights = weights*w_unc
            if self.conf.jacobi_scaling:
                J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # # solve the linear system
            # g, H = self.build_system(J, res, weights)
            # delta = optimizer_step(g, H, lambda_, mask=~failed)
            # if self.conf.jacobi_scaling:
            #     delta = delta * J_scaling

            # # solve the nn optimizer
            delta = self.nnrefine(F_query, F_ref2D, scale)

            if self.conf.pose_from == 'aa':
                # compute the pose update
                # how to rescale the feature?: TODO
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

            # T = T_delta @ T
            if self.conf.range == True:
                shift = (T_delta @ T) @ T_init.inv()
                B = dt.size(0)
                t = shift.t[:, :2]
                rand_t = torch.distributions.uniform.Uniform(-1, 1).sample([B, 2]).to(dt.device)
                rand_t.requires_grad = True
                t = torch.where((t > -10) & (t < 10), t, rand_t)
                zero = torch.zeros([B, 1]).to(t.device)
                # zero = shift.t[:, 2:3]
                t = torch.cat([t, zero], dim=1)
                shift._data[..., -3:] = t
                T = shift @ T_init
            else:
                T = T_delta @ T


            # self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
            #          valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, J=J)
            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, shiftxyr

    # def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
    #          T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
    #          W_ref_query: Optional[Tuple[Tensor, Tensor, int]] = None):
    #
    #     T = T_init
    #     J_scaling = None
    #     if self.conf.normalize_features:
    #         F_query = torch.nn.functional.normalize(F_query, dim=-1)
    #     args = (camera, p3D, F_ref, F_query, W_ref_query)
    #     failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)
    #
    #     lambda_ = self.dampingnet()
    #
    #     for i in range(self.conf.num_iters):
    #         res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
    #
    #         if mask is not None:
    #             valid &= mask
    #         failed = failed | (valid.long().sum(-1) < 10)  # too few points
    #
    #         # compute the cost and aggregate the weights
    #         cost = (res**2).sum(-1)
    #         cost, w_loss, _ = self.loss_fn(cost)
    #         weights = w_loss * valid.float()
    #         if w_unc is not None:
    #             weights = weights*w_unc
    #         if self.conf.jacobi_scaling:
    #             J, J_scaling = self.J_scaling(J, J_scaling, valid)
    #
    #         # solve the linear system
    #         g, H = self.build_system(J, res, weights)
    #         delta = optimizer_step(g, H, lambda_, mask=~failed)
    #         if self.conf.jacobi_scaling:
    #             delta = delta * J_scaling
    #
    #         # compute the pose update
    #         dt, dw = delta.split([3, 3], dim=-1)
    #         # dt, dw = delta.split([2, 1], dim=-1)
    #         # fix z trans, roll and pitch
    #         zeros = torch.zeros_like(dw[:,-1:])
    #         dw = torch.cat([zeros,zeros,dw[:,-1:]], dim=-1)
    #         dt = torch.cat([dt[:,0:2],zeros], dim=-1)
    #
    #         T_delta = Pose.from_aa(dw, dt)
    #         T = T_delta @ T
    #
    #         self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
    #                  valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
    #         if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost):
    #             break
    #
    #     if failed.any():
    #         logger.debug('One batch element had too few valid points.')
    #
    #     return T, failed

class NNrefinev0_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_1, self).__init__()
        self.args = args

        # channel projection
        if self.args.input in ['concat']:
            self.cin = [256, 256, 64]
            self.cout = 256
        else:
            self.cin = [128, 128, 32]
            self.cout = 128

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

        # self.linearp = nn.Sequential(nn.ReLU(inplace=False),
        #                              nn.Linear(3, self.cout),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(self.cout, self.cout))


        if self.args.pool == 'none':
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(self.cout * self.args.max_num_points3D, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())
        # elif self.args.pool == 'aap2':
        #     self.pool = nn.AdaptiveAvgPool1d(4096 // 64)
        #     self.mapping = nn.Sequential(nn.ReLU(inplace=True),
        #                                  nn.Linear(4096 // 64, 1024),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(1024, 32),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(32, 3),
        #                                  nn.Tanh())


    def forward(self, pred_feat, ref_feat, scale, iter=0, level=0):

        B, N, C = pred_feat.size()

        # # normalization
        # if self.args.norm == 'zsn':
        #     pred_feat = (pred_feat - pred_feat.mean()) / (pred_feat.std() + 1e-6)
        #     ref_feat = (ref_feat - ref_feat.mean()) / (ref_feat.std() + 1e-6)
        # else:
        #     pass

        if self.args.input == 'concat':
            r = torch.cat([pred_feat, ref_feat], dim=-1)
        else:
            r = pred_feat - ref_feat  # [B, C, H, W]

        B, N, C = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        # pointfeat = self.linearp(point)
        # x = torch.cat([x, pointfeat], dim=2)
        # x = torch.max(x, 1, keepdim=True)[0]

        if self.args.pool == 'none':
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]

        return y
