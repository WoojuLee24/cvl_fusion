import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa
from pixloc.pixlib.geometry.interpolation import mask_in_image
from pytorch3d.structures import Pointclouds, Volumes
from pytorch3d.ops import add_pointclouds_to_volumes

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
        main_loss='reproj',
        range=False,
        linearp=False,
        attention=False,
        mask=False,
        sat_mask=False,
        input_dim=[128, 128, 32],   # [32, 128, 128],
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

        b, c, a, _ = F_ref.size()

        for i in range(self.conf.num_iters):
            # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
            # res, valid, w_unc, F_ref2D, info = self.cost_fn.residuals(T, *args)

            p3D_ref = T * p3D  # q_3d to q2r_3d
            # p2D, visible = camera.world2image(p3D_ref)
            p3D_ref, valid = camera.world2image3d(p3D_ref)

            # F_p2D_raw, valid, gradients = self.interpolator(F_ref, p2D, return_gradients=False)  # get g2r 2d features
            # valid = mask_in_image(p3D_ref[..., :-1], (a, a), pad=0)
            # valid = valid & visible

            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            if self.conf.mask:
                valid = valid.float().unsqueeze(dim=-1).detach()
                F_query = F_query * valid
                # F_ref2D = F_ref2D * valid
                # p3D = p3D * valid
                p3D_ref = p3D_ref * valid

            # F_q2r_control = self.voxelize(p3D_ref, F_query, size=(b, c, 5, a, a), level=scale)
            F_q2r = camera.voxelize(p3D_ref, F_query, size=(b, c, 20, a, a), level=scale)

            # ## debug
            # from pixloc.visualization.viz_2d import imsave
            #
            # F_q2r_cpu = F_q2r[0].mean(dim=0, keepdim=True)
            # F_ref_cpu = F_ref[0].mean(dim=0, keepdim=True)
            # imsave(F_ref_cpu, '2d_1212', f'{scale}F_ref')
            # imsave(F_q2r_cpu, '2d_1212', f'{scale}F_q2r')
            if self.conf.sat_mask:
                sat_mask = (F_q2r.mean(dim=1, keepdim=True)!=0).float().detach()
                F_ref = F_ref * sat_mask
            delta = self.nnrefine(F_q2r, F_ref, scale)

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
                t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
                zero = torch.zeros([B, 1]).to(t.device)
                # zero = shift.t[:, 2:3]
                t = torch.cat([t, zero], dim=1)
                shift._data[..., -3:] = t
                T = shift @ T_init  # TODO
            else:
                T = T_delta @ T

            # self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
            #          valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, shiftxyr


    # def voxelize(self, xyz, feat, size, level):
    #     B, C, D, A, _ = size
    #     device = feat.device
    #
    #     # R = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=torch.float32, device=feat.device).reshape(3, 3)
    #     #
    #     # if xyz.dim() == 4:
    #     #     zxy = torch.sum(R[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
    #     # elif xyz.dim() == 3:
    #     #     zxy = torch.sum(R[None, None, :, :] * xyz[:, :, None, :], dim=-1)
    #     #
    #     # meter_per_pixel = utils.get_meter_per_pixel() * utils.get_process_satmap_sidelength() / A
    #     # meter_per_vol = torch.tensor([meter_per_pixel, meter_per_pixel, 1], dtype=torch.float32, device=feat.device)
    #     # zxy = zxy / meter_per_vol # + shift
    #
    #     pcs = Pointclouds(points=xyz, features=feat)
    #
    #     init_vol = Volumes(features=torch.zeros(size).to(device),
    #                        densities=torch.zeros((B, 1, D, A, A)).to(device),
    #                        volume_translation=[0, 0, 0],
    #                        )
    #     updated_vol = add_pointclouds_to_volumes(pointclouds=pcs,
    #                                              initial_volumes=init_vol,
    #                                              mode='trilinear',
    #                                              )
    #
    #     features = updated_vol.features()
    #     features = features.mean(dim=2)
    #
    #     return features

class NNrefinev0_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_1, self).__init__()
        self.args = args

        self.cin = self.args.input_dim  # [64, 32, 16] # [128, 128, 32]
        self.cout = 32

        # channel projection
        if self.args.input in ['concat']:
            self.cin = [c*2 for c in self.cin]

        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.linear0 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[0], self.cout, kernel_size=3, stride=1, padding=1))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[1], self.cout, kernel_size=3, stride=1, padding=1))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(self.cin[2], self.cout, kernel_size=3, stride=1, padding=1))

        # if self.args.pool == 'none':
        if self.args.pool == 'aap2':
            self.pooling = nn.AdaptiveAvgPool2d((20, 20))
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout * 20 * 20, 1024),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(1024, 32),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(32, self.yout),
                                         nn.Tanh())
        elif self.args.pool == 'aap3':
            self.pooling = nn.AdaptiveAvgPool2d((40, 40))
            self.mapping = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(self.cout * 40 * 40, 2048),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(2048, 64),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(64, self.yout),
                                         nn.Tanh())


    def forward(self, query_feat, ref_feat, scale):

        B, C, A, _ = query_feat.size()

        if self.args.input == 'concat':     # default
            r = torch.cat([query_feat, ref_feat], dim=1)
        else:
            r = query_feat - ref_feat  # [B, C, H, W]

        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        if 'aap' in self.args.pool:
            x = self.pooling(x)

        x = x.view(B, -1)
        y = self.mapping(x)  # [B, 3]

        return y
