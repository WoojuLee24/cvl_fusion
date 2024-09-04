"""
The top-level model of training-time PixLoc.
Encapsulates the feature extraction, pose optimization, loss and metrics.
"""
import torch
from torch.nn import functional as nnF
import logging
from copy import deepcopy
import omegaconf
import itertools
import numpy as np

from pixloc.pixlib.models.base_model import BaseModel
from pixloc.pixlib.models import get_model
from pixloc.pixlib.models.utils import masked_mean, merge_confidence_map, extract_keypoints
from pixloc.pixlib.geometry.losses import scaled_barron
from pixloc.pixlib.geometry.wrappers import project_grd_to_map, project_map_to_grd
from pixloc.visualization.viz_2d import imsave

from pixloc.visualization.viz_2d import features_to_RGB,plot_images, plot_keypoints
from pixloc.pixlib.utils.tensor import map_tensor
import matplotlib as mpl

from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
import time



logger = logging.getLogger(__name__)

pose_loss = True

def get_weight_from_reproloss(err):
    # the reprojection loss is from 0 to 16.67 ,tensor[B]
    err = err.detach()
    weight = torch.ones_like(err)*err
    weight[err < 10.] = 0
    weight = torch.clamp(weight, min=0., max=50.)

    return weight

class TwoViewRefiner2D(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'unet', #'s2dnet',
        },
        'optimizer': {
            'name': 'nn_optimizer2dv2', # 'learned_optimizer', #'basic_optimizer',
            'input': 'res',
            'pose_loss': False,
            'main_loss': 'reproj2',
            'coe_lat': 1.,
            'coe_lon': 1.,
            'coe_rot': 1.,
            'attention': False,
            'kp': 1.,
            'kd': 1.,
            'ki': 1.,
            'version': 0.1,
        },
        'duplicate_optimizer_per_scale': False,
        'success_thresh': 3,
        'clamp_error': 7777,
        'normalize_features': True,
        'normalize_dt': True,

        # deprecated entries
        'init_target_offset': None,
    }
    required_data_keys = {
        'ref': ['image', 'camera', 'T_w2cam'],
        'query': ['image', 'camera', 'T_w2cam'],
    }
    strict_conf = False  # need to pass new confs to children models

    def _init(self, conf):
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        assert hasattr(self.extractor, 'scales')

        Opt = get_model(conf.optimizer.name)
        if conf.duplicate_optimizer_per_scale:
            oconfs = [deepcopy(conf.optimizer) for _ in self.extractor.scales]
            feature_dim = self.extractor.conf.output_dim
            if not isinstance(feature_dim, int):
                for d, oconf in zip(feature_dim, oconfs):
                    with omegaconf.read_write(oconf):
                        with omegaconf.open_dict(oconf):
                            oconf.feature_dim = d
            self.optimizer = torch.nn.ModuleList([Opt(c) for c in oconfs])
        else:
            self.optimizer = Opt(conf.optimizer)

        if conf.init_target_offset is not None:
            raise ValueError('This entry has been deprecated. Please instead '
                             'use the `init_pose` config of the dataloader.')

    def _forward(self, data):
        def process_siamese(data_i, data_type):
            if data_type == 'ref':
                data_i['type'] = 'sat'
            pred_i = self.extractor(data_i)
            pred_i['camera_pyr'] = [data_i['camera'].scale(1 / s)
                                    for s in self.extractor.scales]
            return pred_i
        pred = {i: process_siamese(data[i], i) for i in ['ref', 'query']}

        # p3D_query = data['query']['points3D']
        T_init = data['T_q2r_init']
        pred['T_q2r_init'] = []
        pred['T_q2r_opt'] = []
        pred['shiftxyr'] = []
        # pred['shiftxyr1'] = []
        # pred['uv_opt'] = []
        # pred['uv_gt'] = []
        pred['pose_loss'] = []

        r2q_img, r2q_mask, p3d_grd, _ = project_map_to_grd(data['T_q2r_gt'], data['query']['camera'].cuda(),
                                                           data['ref']['camera'].cuda(),
                                                           data['query']['image'], data['ref']['image'], data)
        b, c, h, w = data['query']['image'].size()
        data['query']['points3D'] = p3d_grd.detach().reshape(-1, h*w, c)

        if 0:
            r2q_img, r2q_mask, p3d_grd, _ = project_map_to_grd(data['T_q2r_gt'], data['query']['camera'].cuda(), data['ref']['camera'].cuda(),
                                         data['query']['image'], data['ref']['image'], data)

            q2r_img, q2r_mask, _, _ = project_grd_to_map(data['T_q2r_gt'], data['query']['camera'].cuda(), data['ref']['camera'].cuda(),
                                         data['query']['image'], data['ref']['image'], data)

            from pixloc.visualization.viz_2d import imsave
            path = 'debug_images/kitti2' #'visualizations/dense'
            imsave(q2r_img[0], f'/ws/external/{path}', '0q2r')
            imsave(data['query']['image'][0], f'/ws/external/{path}', '0grd')
            imsave(data['ref']['image'][0], f'/ws/external/{path}', '0sat')
            imsave(r2q_img[0], f'/ws/external/{path}', '1r2q')
            imsave(data['query']['image'][0], f'/ws/external/{path}', '1grd')

            imsave(data['ref']['image'][0], f'/ws/external/{path}', '1sat')
            # print(f"roll: {data['roll']}, pitch: {data['pitch']}")

        for i in reversed(range(len(self.extractor.scales))):
            if self.conf.optimizer.attention:
                F_ref = pred['ref']['feature_maps'][i] * pred['ref']['confidences'][i]
            else:
                F_ref = pred['ref']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]

            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
            else:
                opt = self.optimizer

            if self.conf.optimizer.attention:
                F_q = pred['query']['feature_maps'][i] * pred['query']['confidences'][i]
            else:
                F_q = pred['query']['feature_maps'][i]
            cam_q = pred['query']['camera_pyr'][i]

            F_r2q, mask_r2q, p3d_grd, p3d_g2s = project_map_to_grd(T_init, cam_q, cam_ref, F_q, F_ref, data)
            F_q2r, mask_q2r, p3d_s, p3d_s2g = project_grd_to_map(T_init, cam_q, cam_ref, F_q, F_ref, data)

            if 0:
                F_r2q, mask_r2q, p3d_grd, p3d_g2s = project_map_to_grd(data['T_q2r_gt'], cam_q, cam_ref, F_q, F_ref, data)
                F_q2r, mask_q2r, p3d_s, p3d_s2g = project_grd_to_map(data['T_q2r_gt'], cam_q, cam_ref, F_q, F_ref, data)

                from pixloc.visualization.viz_2d import imsave
                imsave(F_q2r[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '2q2rf')
                imsave(F_q[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '2grdf')
                imsave(F_ref[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '2satf')
                imsave(F_r2q[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '3r2qf')
                imsave(F_q[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '3grdf')
                imsave(F_ref[0].mean(dim=0, keepdim=True), '/ws/external/visualizations/dense', '3satf')


            # # p3D_query2 = grd_img2cam(cam_q, F_q.size(-2), F_q.size(-1), 375, 1242)
            # p2D_query, visible = cam_q.world2image(data['query']['T_w2cam']*p3D_query)
            # F_q_key, mask, _ = opt.interpolator(F_q, p2D_query)
            # mask &= visible

            if self.conf.optimizer.jacobian:
                W_q = pred['query']['confidences'][i]
                W_q, _, _ = opt.interpolator(W_q, p2D_query)
                W_ref = pred['ref']['confidences'][i]
                W_ref_q = (W_ref, W_q, 1)
            else:
                W_ref_q = None

            # W_q = pred['query']['confidences'][i]
            # W_q, _, _ = opt.interpolator(W_q, p2D_query)
            # W_ref = pred['ref']['confidences'][i]
            # W_ref_q = (W_ref, W_q, 1)

            if self.conf.normalize_features in ['l2', True]: # B x C x H X W
                F_q = nnF.normalize(F_q, dim=1)
                F_ref = nnF.normalize(F_ref, dim=1)
                F_q2r = nnF.normalize(F_q2r, dim=1)
                F_r2q = nnF.normalize(F_r2q, dim=1)


            ### Pose estimator ###
            T_opt, failed, shiftxyr = opt(
                dict(p3D=p3d_grd, F_ref=F_ref, F_q=F_q, F_q2r=F_q2r, F_r2q=F_r2q,
                T_init=T_init, cam_ref=cam_ref, cam_q=cam_q,
                mask_q2r=mask_q2r, mask_r2q=mask_r2q, W_ref_q=W_ref_q, data=data, scale=i, version=self.conf.optimizer.version))


            pred['T_q2r_init'].append(T_init)
            pred['T_q2r_opt'].append(T_opt)
            pred['shiftxyr'].append(shiftxyr)

            if self.conf.optimizer.cascade:
                T_init = T_opt
            else:
                T_init = T_opt.detach()

        return pred

    def project_grd_to_map2(self, T, cam_q, F_query, F_ref):
        # g2s with GH and T
        b, c, a, a = F_ref.size()
        b, c, h, w = F_query.size()
        uv1 = self.get_warp_sat2real(F_ref)
        uv, mask = self.seq_warp_real2camera(T, cam_q, uv1)

        # uv1 = uv1.reshape(-1, 3).repeat(b, 1, 1).contiguous()
        # uv1 = T.cuda().inv() * uv1
        # uv, mask = cam_q.world2image(uv1)
        # uv = uv.reshape(b, a, a, 2).contiguous()

        scale = torch.tensor([w - 1, h - 1]).to(uv)
        uv = (uv / scale) * 2 - 1
        uv = uv.clamp(min=-2, max=2)  # ideally use the mask instead
        F_g2s = torch.nn.functional.grid_sample(F_query, uv, mode='bilinear', align_corners=True)

        return F_g2s

    def seq_warp_real2camera(self, T_q2r, cam_q, uv1):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        XYZ_1 = torch.cat([uv1, torch.ones_like(uv1[..., 0:1])], dim=-1)
        T_r2q = T_q2r.cuda().inv()
        R = T_r2q.R  # shape = [B,3,3]
        T = T_r2q.t.unsqueeze(dim=-1)

        B = R.shape[0]

        camera_height = 1.65
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        # height = camera_height * torch.ones_like([B, 1])
        # T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        # T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        zeros = torch.zeros_like(cam_q.f[:, 0:1])
        ones = torch.ones_like(cam_q.f[:, 0:1])
        camera_k = torch.cat([cam_q.f[:, 0:1], zeros, cam_q.c[:, 0:1],
                              zeros, cam_q.f[:, 1:2], cam_q.c[:, 1:2],
                              zeros, zeros, ones], dim=-1)
        camera_k = camera_k.view(B, 3, 3)
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        mask = torch.squeeze(mask, dim=-1)
        return uv, mask


    def project_polar_to_grid(self, uv, T, cam_q, F_query, F_ref):
        # g2s with GH and T
        b, c, a, a = F_ref.size()
        b, c, h, w = F_query.size()

        f, c = cam_q.f[..., 0][..., None, None], cam_q.c[..., 0][..., None, None]
        u = uv[..., 0] / (uv[..., -1] + 1e-8) * f # + c
        z_idx = uv[..., -1]
        z_idx = torch.arange(1279, -1, -1, dtype=torch.float32).unsqueeze(1).repeat(1,  1280).repeat(2, 1, 1).cuda()
        uv_grid = torch.stack([u, z_idx], -1)

        # uv1 = self.get_warp_sat2real(F_ref)
        #
        # # uv, mask = cam_query.world2image(uv1)
        # uv1 = uv1.reshape(-1, 3).repeat(b, 1, 1).contiguous()
        # uv1 = T.cuda().inv() * uv1
        #
        # uv, mask = cam_q.world2image(uv1)
        # uv = uv.reshape(b, a, a, 2).contiguous()
        #
        # # scale = torch.tensor([w - 1, h - 1]).to(uv)
        scale = torch.tensor([w - 1, h - 1]).to(uv_grid)
        uv_grid = (uv_grid / scale) * 2 - 1
        uv_grid = uv_grid.clamp(min=-2, max=2)  # ideally use the mask instead
        F_g2s = torch.nn.functional.grid_sample(F_query, uv_grid, mode='bilinear', align_corners=True)

        return F_g2s

    def project_grd_to_map(self, T, cam_q, cam_ref, F_query, F_ref, meter_per_pixel=0.078302836):
        # g2s with GH and T
        b, c, a, a = F_ref.size()
        b, c, h, w = F_query.size()

        uv1 = self.get_warp_sat2real(cam_ref, F_ref, meter_per_pixel)
        # uv1 = uv1.reshape(-1, 3).repeat(b, 1, 1).contiguous()
        uv1 = uv1.reshape(b, -1, 3).contiguous()

        uv1 = T.cuda().inv() * uv1

        uv, mask = cam_q.world2image(uv1)

        uv = uv.reshape(b, a, a, 2).contiguous()
        mask = mask.reshape(b, a, a, 1).contiguous().float()

        scale = torch.tensor([w - 1, h - 1]).to(uv)
        uv = (uv / scale) * 2 - 1
        uv = uv.clamp(min=-2, max=2)  # ideally use the mask instead

        F_g2s = torch.nn.functional.grid_sample(F_query, uv, mode='bilinear', align_corners=True)

        return F_g2s, uv

    def get_warp_sat2real(self, cam_ref, F_ref, meter_per_pixel):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        B, C, _, satmap_sidelength = F_ref.size()

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        # u0 = v0 = satmap_sidelength // 2
        # uv_center = uv - torch.tensor(
        #     [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]
        center = cam_ref.c
        uv_center = uv.repeat(B, 1, 1, 1) - center.unsqueeze(dim=1).unsqueeze(
            dim=1)  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # inv equation (1)
        # meter_per_pixel = 0.07463721  # 0.298548836 / 5 # 0.298548836 (paper) # 0.07463721(1280) #0.1958(512) 0.078302836
        meter_per_pixel *= 1280 / satmap_sidelength
        # R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        # Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        XY = uv_center * meter_per_pixel
        Z = torch.zeros_like(XY[..., 0:1])
        ones = torch.ones_like(Z)
        # sat2realwap = torch.cat([XY[:, :, :1], Z, XY[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]
        XYZ = torch.cat([XY[..., :1], XY[..., 1:], Z], dim=-1)  # [sidelength,sidelength,4]
        # XYZ = XYZ.unsqueeze(dim=0)
        # XYZ = XYZ.reshape(B, -1, 3)

        return XYZ

    def preject_l1loss(self, opt, p3D, F_ref, F_query, T_gt, camera, mask=None, W_ref_query= None):
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        res, valid, w_unc, _, _ = opt.cost_fn.residuals(T_gt, *args)
        if mask is not None:
            valid &= mask

        # compute the cost and aggregate the weights
        cost = (res ** 2).sum(-1)
        cost, w_loss, _ = opt.loss_fn(cost) # robust cost
        loss = cost * valid.float()
        if w_unc is not None:
            loss = loss * w_unc

        return torch.sum(loss, dim=-1)/(torch.sum(valid)+1e-6)

    def add_grd_confidence(self):
        self.extractor.add_grd_confidence()

    def loss(self, pred, data):
        if self.conf.optimizer.main_loss == 'rt':
            losses = self.rt_loss(pred, data)
        elif self.conf.optimizer.main_loss == "reproj2":
            losses = self.reproj_loss2(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'rtreproj':
            losses = self.rtreproj_loss(pred, data)  # rtreproj
        elif self.conf.optimizer.main_loss == 'reproj_rgb':
            losses = self.reproj_rgb_loss(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'reproj_fusion':
            losses = self.reproj_fusion_loss(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'reprojx2':
            losses = self.reproj_lossx2(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'metric':
            losses = self.metric_loss(pred, data)
        else:
            losses = self.reproj_loss(pred, data)  # default = reproj

        return losses


    def reproj_loss2(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            # err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        losses = {'total': 0.}

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses


    def reproj_lossx2(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0

        # shift1
        for i, T_opt in enumerate(pred['shift1']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            success = err < thresh
            losses[f'reprojection1_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss:
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss:
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses

    def reproj_rgb_loss(self, pred, data):
        B = data['ref']['image'].size(0)
        num_scales = len(self.extractor.scales)
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0

        def reprojection_error(uv, uv_gt):
            uv_diff = uv - uv_gt
            uv_diff = uv_diff.reshape(B, -1)
            uv_mse = torch.sqrt(torch.mean(torch.square(uv_diff), dim=-1))
            return uv_mse

        uv_gt = project_grd_to_map(data['T_q2r_gt'],
                                   data['query']['camera'], data['ref']['camera'],
                                   data['query']['image'], data['ref']['image'],
                                   meter_per_pixel=0.078302836
                                   )

        uv_init = project_grd_to_map(data['T_q2r_init'],
                                     data['query']['camera'], data['ref']['camera'],
                                     data['query']['image'], data['ref']['image'],
                                     meter_per_pixel=0.078302836
                                     )

        rgb_err_init = reprojection_error(uv_init, uv_gt)
        losses[f'reprojection_rgb_error/init'] = rgb_err_init

        # RGB proj
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            uv = project_grd_to_map(T_opt,
                                    data['query']['camera'], data['ref']['camera'],
                                    data['query']['image'], data['ref']['image'],
                                    meter_per_pixel=0.078302836
                                    )
            uv_mse = reprojection_error(uv, uv_gt)
            rgb_err = uv_mse / num_scales
            losses[f'reprojection_rgb_error/{i}'] = rgb_err
            losses['total'] += self.conf.optimizer.coe_rot * rgb_err

        losses['reprojection_error'] = rgb_err

        return losses


    def reproj_fusion_loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']
        B = pred['uv_opt'][0].size(0)

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0

        # RGB proj
        for i, uv_opt in enumerate(pred['uv_opt']):
            uv_diff = uv_opt - pred['uv_gt'][i]
            uv_diff = uv_diff.reshape(B, -1)
            uv_mse = torch.sqrt(torch.mean(torch.square(uv_diff), dim=-1))
            rgb_err = uv_mse / num_scales
            losses[f'reprojection_rgb_error/{i}'] = rgb_err
            losses['total'] += self.conf.optimizer.coe_rot * rgb_err

        # LiDAR proj
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses

    def rtreproj_loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']
        shift_gt = data['shift_gt']
        shift_init = torch.zeros_like(shift_gt)
        coe = torch.tensor([[self.conf.optimizer.coe_lat,
                             self.conf.optimizer.coe_lon,
                             self.conf.optimizer.coe_rot]]).to(shift_init.device)

        def shift_error(shift):
            err = torch.abs(shift - shift_gt)
            # err = scaled_barron(1., 2.)(err)[0] / 4
            # err = err.mean(dim=0, keepdim=True)
            return err

        err_init = shift_error(shift_init)
        num_scales = len(self.extractor.scales)
        # success = None
        losses = {'total': 0.}

        for i, shift in enumerate(pred['shiftxyr1']):
            err = shift_error(shift)
            err_lat = err[:, 0].detach()
            err_lon = err[:, 1].detach()
            err_rot = err[:, 2].detach()
            loss = (coe * err).sum(dim=-1) / num_scales

            losses[f'error/{i}'] = err.mean(dim=-1).detach()
            losses[f'error_lat/{i}'] = err_lat
            losses[f'error_lon/{i}'] = err_lon
            losses[f'error_rot/{i}'] = err_rot

            losses['total'] += loss

        losses['error'] = err.mean(dim=-1).detach()
        losses['error_init'] = err_init.mean(dim=-1).detach()

        reproj_losses = self.reproj_loss(pred, data)
        losses['reprojection_error'] = reproj_losses['reprojection_error']
        losses['total'] += reproj_losses['reprojection_error']

        return losses

    def rt_loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']
        shift_gt = data['shift_gt']
        shift_init = torch.zeros_like(shift_gt)
        coe = torch.tensor([[self.conf.optimizer.coe_lat,
                             self.conf.optimizer.coe_lon,
                             self.conf.optimizer.coe_rot]]).to(shift_init.device)

        def shift_error(shift):
            err = torch.abs(shift - shift_gt)
            # err = scaled_barron(1., 2.)(err)[0] / 4
            # err = err.mean(dim=0, keepdim=True)
            return err

        err_init = shift_error(shift_init)
        num_scales = len(self.extractor.scales)
        losses = {'total': 0.}

        for i, shift in enumerate(pred['shiftxyr']):
            err = shift_error(shift)
            err_lat = err[:, 0].detach()
            err_lon = err[:, 1].detach()
            err_rot = err[:, 2].detach()
            loss = (coe * err).sum(dim=-1) / num_scales

            losses[f'error/{i}'] = err.mean(dim=-1).detach()
            losses[f'error_lat/{i}'] = err_lat
            losses[f'error_lon/{i}'] = err_lon
            losses[f'error_rot/{i}'] = err_rot

            losses['total'] += loss

        losses['error'] = err.mean(dim=-1).detach()
        losses['error_init'] = err_init.mean(dim=-1).detach()

        with torch.no_grad():
            reproj_losses = self.reproj_loss(pred, data)
            losses['reprojection_error'] = reproj_losses['reprojection_error'].detach()

        return losses

    def reproj_loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            success = err < thresh

            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss:
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        # ## logging ##
        # with torch.no_grad():
        #     for i, shiftxyr in enumerate(pred['shiftxyr']):
        #         shift_error = torch.abs(data['shift_gt'] - shiftxyr)
        #         losses[f'lat_error/{i}'] = shift_error[..., 0]
        #         losses[f'lon_error/{i}'] = shift_error[..., 1]
        #         losses[f'rot_error/{i}'] = shift_error[..., 2]
        #     if pred['shiftxyr1'] is not None:
        #         for i, shiftxyr1 in enumerate(pred['shiftxyr1']):
        #             shift_error = torch.abs(data['shift_gt'] - shiftxyr1)
        #             losses[f'lat_error1/{i}'] = shift_error[..., 0]
        #             losses[f'lon_error1/{i}'] = shift_error[..., 1]
        #             losses[f'rot_error1/{i}'] = shift_error[..., 2]

        return losses

    def reprojrt_loss(self, pred, data):
        pass


    def metric_loss(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()
        num_scales = len(self.extractor.scales)

        def scaled_pose_error(T_q2r):
            # err_R, err_t = (T_r2q_gt @ T_q2r).magnitude()
            # err_lat, err_long = (T_r2q_gt @ T_q2r).magnitude_latlong()
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            return err_R, err_t, err_lat, err_long

        metrics = {'total': 0.}
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = scaled_pose_error(T_opt)
            loss = (err[0] + err[1]).mean(dim=-1, keepdim=True)
            metrics['total'] += loss / num_scales

            metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[
                f'long_error/{i}'] = err
        metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error'] = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[
            f'long_error/init'] = err_init

        # with torch.no_grad():
        #     reproj_losses = self.reproj_loss(pred, data)
        #     metrics['reprojection_error'] = reproj_losses['reprojection_error'].detach()

        return metrics


    def metrics(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_q2r):
            # err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
            # err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            return err_R, err_t, err_lat, err_long

        metrics = {}
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[f'long_error/{i}'] = err
        metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error']  = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[f'long_error/init'] = err_init

        return metrics
