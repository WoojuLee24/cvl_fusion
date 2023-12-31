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


class NNOptimizer2D3D(BaseOptimizer):
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
        linearp=False,
        attention=False,
        mask=True,
        input_dim=[128, 128, 32],  # [32, 128, 128],
        pool_rgb='none',
        range=True,  # 'none',
        mode=1, # 2
        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.conf = conf
        self.dampingnet = DampingNet(conf.damping)
        self.nnrefine = NNrefinev0_1(conf)
        self.nnrefine_rgb = NNrefinev0_2_1(conf)
        assert conf.learned_damping
        super()._init(conf)


    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['F_q_key'], data['T_init'],
            data['cam_ref'], data['cam_q'], data['mask'], data.get('W_ref_q'), data,
            data['scale'], data['mode'])


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor, F_q_key: Tensor,
             T_init: Pose, cam_ref: Camera, cam_q: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor, int]] = None,
             data=None,
             scale=None,
             mode='rgb'):

        T = T_init
        shift_gt = data['data']['shift_gt']
        shift_range = data['data']['shift_range']

        J_scaling = None
        if self.conf.normalize_features:
            F_q_key = torch.nn.functional.normalize(F_q_key, dim=-1)
        args = (cam_ref, p3D, F_ref, F_q_key, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()
        shiftxyr = torch.zeros_like(shift_range)
        shift1 = None

        if mode == 1:
            # RGB
            for i in range(self.conf.num_iters):
                F_g2s = self.project_grd_to_map(T, cam_q, F_query, F_ref)

                # save_path = '3d_1226'
                # from pixloc.visualization.viz_2d import imsave
                # imsave(F_g2s[0].mean(dim=0, keepdim=True), save_path, f'fg2s_{scale}')

                # # solve the nn optimizer
                delta = self.nnrefine_rgb(F_g2s, F_ref, scale)

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
                T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)


                # fusion
                # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
                res, valid, w_unc, F_ref_key, info = self.cost_fn.residuals(T, *args)
                p3D_ref = T * p3D

                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                if self.conf.mask:
                    valid = valid.float().unsqueeze(dim=-1).detach()
                    F_q_key = F_q_key * valid
                    F_ref_key = F_ref_key * valid
                    p3D = p3D * valid
                    p3D_ref = p3D_ref * valid

                # # solve the nn optimizer
                delta = self.nnrefine(F_q_key, F_ref_key, p3D, p3D_ref, scale)

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

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

        elif mode == 2:
            # RGB
            for i in range(self.conf.num_iters):
                F_g2s = self.project_grd_to_map(T, cam_q, F_query, F_ref)

                # save_path = '3d_1226'
                # from pixloc.visualization.viz_2d import imsave
                # imsave(F_g2s[0].mean(dim=0, keepdim=True), save_path, f'fg2s_{scale}')

                # # solve the nn optimizer
                delta = self.nnrefine_rgb(F_g2s, F_ref, scale)

                # rescaling
                delta = delta * shift_range
                delta_mask = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(delta.device)
                delta = delta * delta_mask
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]
                dt = torch.cat([zeros, zeros, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)
                T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

                # fusion
                # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
                res, valid, w_unc, F_ref_key, info = self.cost_fn.residuals(T, *args)
                p3D_ref = T * p3D

                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                if self.conf.mask:
                    valid = valid.float().unsqueeze(dim=-1).detach()
                    F_q_key = F_q_key * valid
                    F_ref_key = F_ref_key * valid
                    p3D = p3D * valid
                    p3D_ref = p3D_ref * valid

                # # solve the nn optimizer
                delta = self.nnrefine(F_q_key, F_ref_key, p3D, p3D_ref, scale)

                # rescaling
                delta = delta * shift_range
                delta_mask = torch.tensor([[1, 1, 0]], dtype=torch.float32).to(delta.device)
                delta = delta * delta_mask
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                dw = torch.zeros_like(dw)
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
                    t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
                    zero = torch.zeros([B, 1]).to(t.device)
                    # zero = shift.t[:, 2:3]
                    t = torch.cat([t, zero], dim=1)
                    shift._data[..., -3:] = t
                    T = shift @ T_init  # TODO
                else:
                    T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

        elif mode == 3:
            # RGB
            for i in range(self.conf.num_iters):
                F_g2s = self.project_grd_to_map(T, cam_q, F_query, F_ref)

                # save_path = '3d_1226'
                # from pixloc.visualization.viz_2d import imsave
                # imsave(F_g2s[0].mean(dim=0, keepdim=True), save_path, f'fg2s_{scale}')

                # # solve the nn optimizer
                delta = self.nnrefine_rgb(F_g2s, F_ref, scale)

                # rescaling
                delta = delta * shift_range
                delta_mask = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(delta.device)
                delta = delta * delta_mask
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]
                dt = torch.cat([zeros, zeros, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)
                T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

                # fusion
                # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
                res, valid, w_unc, F_ref_key, info = self.cost_fn.residuals(T, *args)
                p3D_ref = T * p3D

                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                if self.conf.mask:
                    valid = valid.float().unsqueeze(dim=-1).detach()
                    F_q_key = F_q_key * valid
                    F_ref_key = F_ref_key * valid
                    p3D = p3D * valid
                    p3D_ref = p3D_ref * valid

                # # solve the nn optimizer
                delta = self.nnrefine(F_q_key, F_ref_key, p3D, p3D_ref, scale)

                # rescaling
                delta = delta * shift_range
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                dw = torch.zeros_like(dw)
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
                    t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
                    zero = torch.zeros([B, 1]).to(t.device)
                    # zero = shift.t[:, 2:3]
                    t = torch.cat([t, zero], dim=1)
                    shift._data[..., -3:] = t
                    T = shift @ T_init  # TODO
                else:
                    T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

        elif mode == 4:
            """ shift1 is shift error for rtreproj loss"""
            shift1 = torch.zeros_like(shift_range)
            # RGB
            for i in range(self.conf.num_iters):
                F_g2s = self.project_grd_to_map(T, cam_q, F_query, F_ref)

                # save_path = '3d_1226'
                # from pixloc.visualization.viz_2d import imsave
                # imsave(F_g2s[0].mean(dim=0, keepdim=True), save_path, f'fg2s_{scale}')

                # # solve the nn optimizer
                delta = self.nnrefine_rgb(F_g2s, F_ref, scale)

                # rescaling
                delta = delta * shift_range
                # delta_mask = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(delta.device)
                # delta = delta * delta_mask
                shift1 += delta
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]
                dt = torch.cat([zeros, zeros, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)
                T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

                # fusion
                # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
                res, valid, w_unc, F_ref_key, info = self.cost_fn.residuals(T, *args)
                p3D_ref = T * p3D

                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                if self.conf.mask:
                    valid = valid.float().unsqueeze(dim=-1).detach()
                    F_q_key = F_q_key * valid
                    F_ref_key = F_ref_key * valid
                    p3D = p3D * valid
                    p3D_ref = p3D_ref * valid

                # # solve the nn optimizer
                delta = self.nnrefine(F_q_key, F_ref_key, p3D, p3D_ref, scale)

                # rescaling
                delta = delta * shift_range
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                dw = torch.zeros_like(dw)
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
                    t = torch.where((t > -shift_range[0][0]) & (t < shift_range[0][0]), t, rand_t)
                    zero = torch.zeros([B, 1]).to(t.device)
                    # zero = shift.t[:, 2:3]
                    t = torch.cat([t, zero], dim=1)
                    shift._data[..., -3:] = t
                    T = shift @ T_init  # TODO
                else:
                    T = T_delta @ T

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

        elif mode == 5:
            """ shift1 is T1 for reprojx2 loss"""
            # RGB
            for i in range(self.conf.num_iters):
                F_g2s = self.project_grd_to_map(T, cam_q, F_query, F_ref)

                # save_path = '3d_1226'
                # from pixloc.visualization.viz_2d import imsave
                # imsave(F_g2s[0].mean(dim=0, keepdim=True), save_path, f'fg2s_{scale}')

                # # solve the nn optimizer
                delta = self.nnrefine_rgb(F_g2s, F_ref, scale)

                # rescaling
                delta = delta * shift_range
                # delta_mask = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(delta.device)
                # delta = delta * delta_mask
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]
                dt = torch.cat([zeros, zeros, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)
                T1 = T_delta @ T
                shift1 = T1

                self.log(i=i, T_init=T_init, T=T1, T_delta=T_delta)

                # fusion
                # res, valid, w_unc, F_ref2D, J = self.cost_fn.residual_jacobian(T, *args)
                res, valid, w_unc, F_ref_key, info = self.cost_fn.residuals(T1, *args)
                p3D_ref = T1 * p3D

                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

                if self.conf.mask:
                    valid = valid.float().unsqueeze(dim=-1).detach()
                    F_q_key = F_q_key * valid
                    F_ref_key = F_ref_key * valid
                    p3D = p3D * valid
                    p3D_ref = p3D_ref * valid

                # # solve the nn optimizer
                delta = self.nnrefine(F_q_key, F_ref_key, p3D, p3D_ref, scale)

                # rescaling
                delta = delta * shift_range
                shiftxyr += delta

                dt, dw = delta.split([2, 1], dim=-1)
                B = dw.size(0)

                dw = torch.zeros_like(dw)
                cos = torch.cos(dw)
                sin = torch.sin(dw)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                dR = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B,9]
                dR = dR.view(B, 3, 3)  # shape = [B,3,3]

                dt = torch.cat([dt, zeros], dim=-1)

                T_delta = Pose.from_Rt(dR, dt)

                if self.conf.range == True:
                    shift = (T_delta @ T1) @ T_init.inv()
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
                    T = T_delta @ T1

                self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)


        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, shiftxyr, shift1

    def project_grd_to_map(self, T, cam_q, F_query, F_ref):
        # g2s with GH and T
        b, c, a, a = F_ref.size()
        b, c, h, w = F_query.size()
        uv1 = self.get_warp_sat2real(F_ref)
        uv1 = uv1.reshape(-1, 3).repeat(b, 1, 1).contiguous()
        uv1 = T.cuda().inv() * uv1
        # uv, mask = cam_query.world2image(uv1)
        uv, mask = cam_q.world2image(uv1)
        uv = uv.reshape(b, a, a, 2).contiguous()

        scale = torch.tensor([w - 1, h - 1]).to(uv)
        uv = (uv / scale) * 2 - 1
        uv = uv.clamp(min=-2, max=2)  # ideally use the mask instead
        F_g2s = torch.nn.functional.grid_sample(F_query, uv, mode='bilinear', align_corners=True)

        return F_g2s

    def get_warp_sat2real(self, F_ref):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        B, C, _, satmap_sidelength = F_ref.size()

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # inv equation (1)
        meter_per_pixel = 0.07463721  # 0.298548836 / 5 # 0.298548836 (paper) # 0.07463721(1280) #0.1958(512)
        meter_per_pixel *= 1280 / satmap_sidelength
        # R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        # Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        XY = uv_center * meter_per_pixel
        Z = torch.zeros_like(XY[..., 0:1])
        ones = torch.ones_like(Z)
        # sat2realwap = torch.cat([XY[:, :, :1], Z, XY[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]
        XYZ = torch.cat([XY[:, :, :1], XY[:, :, 1:], Z], dim=-1)  # [sidelength,sidelength,4]
        # XYZ = XYZ.unsqueeze(dim=0)
        # XYZ = XYZ.reshape(B, -1, 3)

        return XYZ

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
        elif 'embed' in self.args.pool:
            x = x.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
        elif self.args.pool == 'avg':
            x = torch.mean(x, 1, keepdim=True)

        # if self.args.pool == 'none':
        x = x.view(B, -1)
        y = self.mapping(x)  # [B, 3]

        return y


class NNrefinev0_2_1(nn.Module):
    def __init__(self, args):
        super(NNrefinev0_2_1, self).__init__()
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


    def forward(self, pred_feat, ref_feat, scale):

        B, C, _, _ = pred_feat.size()

        # normalization
        if self.args.norm == 'zsn':
            pred_feat = (pred_feat - pred_feat.mean()) / (pred_feat.std() + 1e-6)
            ref_feat = (ref_feat - ref_feat.mean()) / (ref_feat.std() + 1e-6)
        else:
            pass

        if self.args.input == 'concat':     # default
            r = torch.cat([pred_feat, ref_feat], dim=-1)
        else:
            r = pred_feat - ref_feat  # [B, C, H, W]

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
        elif 'aap' in self.args.pool_rgb:
            x = self.pool(x)
            B, C, H, W = x.size()
            x = x.view(B, C*H*W)
            y = self.mapping(x)  # [B, 3]

        return y

