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


class NNOptimizer2D(BaseOptimizer):
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
        range=False,
        ref_key=False,
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
            data['camera'], data['mask'], data.get('W_ref_q'), data['p2D'], data['scale'])


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor, int]] = None,
             p2D = None,
             scale = None):

        T = T_init

        J_scaling = None
        if self.conf.normalize_features:
            F_query = torch.nn.functional.normalize(F_query, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()

        for i in range(self.conf.num_iters):
            # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
            #
            # if mask is not None:
            #     valid &= mask
            # failed = failed | (valid.long().sum(-1) < 10)  # too few points
            #
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


            p3D_r = T * p3D  # q_3d to q2r_3d
            p2D_r, visible = camera.world2image(p3D_r)  # q2r_3d to q2r_2d
            b, c, h, w = F_ref.size()
            p2D_r_int = torch.round(p2D_r).long()

            B, C, H, W= F_query.size()
            # p2D_int = torch.round(p2D).long()
            F_q2r = torch.zeros_like(F_ref)

            F_query_key, valid, gradients = self.cost_fn.interpolator(F_query,
                                                                      p2D,
                                                                      return_gradients=True)  # get key feature from p3D

            # F_q2r[:, :, p2D_r_int[0, :, 0], p2D_r_int[0, :, 1]] (1, 128, 1024)
            # TODO: F_q2r[p2D_r_int] = F_query[p2D_int] # q2r_2d from query feature
            # get batched_index
            B, N, _ = p2D_r_int.size()
            p2D_r_int = p2D_r_int.view(-1, 2)
            xidx, yidx = p2D_r_int[:, 0], p2D_r_int[:, 1]
            bidx = torch.repeat_interleave(torch.arange(B), N, dim=0)
            F_q2r[bidx, :, yidx, xidx] = F_query_key.contiguous().view(-1, C)
            F_q2r = F_q2r.contiguous()

            if self.conf.ref_key == True:
                # F_ref_key, valid, gradients = self.cost_fn.interpolator(F_ref,
                #                                                         p2D_r,
                #                                                         return_gradients=True)  # get key feature from p3D
                F_ref_key = torch.zeros_like(F_ref)
                F_ref_key[bidx, :, yidx, xidx] = F_ref[bidx, :, yidx, xidx]
                F_ref_key = F_ref_key.contiguous()
            else:
                F_ref_key = F_ref

            # # DEBUG
            # from pixloc.visualization.viz_2d import imsave
            # F_ref_key_mean = F_ref_key.mean(dim=1)
            # F_q2r_mean = F_q2r.mean(dim=1)
            # F_ref_mean = F_ref.mean(dim=1)
            # F_query_mean = F_query.mean(dim=1)
            # imsave(F_ref_mean, "nn2d", "F_ref")
            # imsave(F_query_mean, "nn2d", "F_query")
            # imsave(F_ref_key_mean, "nn2d", "F_ref_key")
            # imsave(F_q2r_mean, "nn2d", "F_q2r")

            # # solve the nn optimizer
            delta = self.nnrefine(F_q2r, F_ref_key, p3D, scale)

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
            # self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
            #          valid=valid, w_unc=w_unc, w_loss=w_loss, J=J)
            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta) #valid=valid)
            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed

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
            self.cout = 128
        else:
            self.cin = [128, 128, 32]
            self.cout = 128

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

        if self.args.pool == 'aap2':
            self.pool = nn.AdaptiveAvgPool2d((20, 20))
            self.mapping = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Linear(self.cout * 20 * 20, 2048),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(2048, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())


    def forward(self, query_feat, ref_feat, point, scale=0, iter=0, level=0):

        B, C, H, W = query_feat.size()

        if self.args.input == 'concat':
            r = torch.cat([query_feat, ref_feat], dim=1)
        else:
            r = query_feat - ref_feat  # [B, C, H, W]

        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        if self.args.pool == 'none':
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]
        elif 'aap' in self.args.pool:
            x = self.pool(x)
            x = x.view(B, -1)
            y = self.mapping(x)

        return y
