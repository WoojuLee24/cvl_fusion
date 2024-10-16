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
from pixloc.pixlib.geometry.wrappers import Camera, Pose
from pixloc.pixlib.models.nn_optimizer3dv2_6 import GridIndexProcessor
from pixloc.visualization.viz_2d import features_to_RGB,plot_images,plot_keypoints
from pixloc.pixlib.utils.tensor import map_tensor
from pixloc.pixlib.models.geo_encoder import DenseEncoder, SparseEncoder
import matplotlib as mpl

from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
import time


logger = logging.getLogger(__name__)

def get_weight_from_reproloss(err):
    # the reprojection loss is from 0 to 16.67 ,tensor[B]
    err = err.detach()
    weight = torch.ones_like(err)*err
    weight[err < 10.] = 0
    weight = torch.clamp(weight, min=0., max=50.)

    return weight

class TwoViewRefiner3D(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'unet', #'s2dnet',
        },
        'geo_encoder': 'none',
        'optimizer': {
            'name': 'nn_optimizer3dv2_6', # 'learned_optimizer', #'basic_optimizer',
            'input': 'res',
            'pose_loss': 'triplet', # 'rr', 'none'
            'pose_lambda': 1,
            'main_loss': 'reproj',
            'coe_lat': 1.,
            'coe_lon': 1.,
            'coe_rot': 1.,
            'cascade': False,
            'attention': False,
            'jacobian': False,
            'multi_pose': 1,
            'max_num_points3D': 5000,
            'max_num_out_points3D': 15000,
            'max_num_features': 15000,
            'voxel_shape': [400, 400, 30],
            'max_volume_space': [100, 100, 10],
            'min_volume_space': [-100, -100, -5],
            'stride': [1, 1],
        },
        'duplicate_optimizer_per_scale': False,
        'success_thresh': 3,
        'clamp_error': 7777,
        'normalize_features': True,
        'normalize_dt': True,
        'debug': False,
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

        # self.grid_processor = GridIndexProcessor(self.args.voxel_shape,
        #                                          self.args.max_volume_space,
        #                                          self.args.min_volume_space)
        if conf.geo_encoder == 'dense':
            self.geo_encoder = DenseEncoder(cout=self.conf.extractor.output_dim[0],
                                            normalize=self.conf.normalize_features)
        elif conf.geo_encoder in ['sp2d_p', 'sp2d_pxyz', 'sp2d_pz', 'sp2d_pza']:
            if conf.geo_encoder == 'sp2d_p':
                cin = 1
            elif conf.geo_encoder == 'sp2d_pxyz':
                cin = 4
            elif conf.geo_encoder == 'sp2d_pz':
                cin = 2
            elif conf.geo_encoder == 'sp2d_pza':
                cin = 3
            self.geo_encoder = SparseEncoder(cin=cin,
                                             cout=self.conf.extractor.output_dim[0],
                                             mode=conf.geo_encoder,
                                             max_num_features=self.conf.optimizer.max_num_features)


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

        p3D_query = data['query']['points3D']
        T_init = data['T_q2r_init']
        pred['T_q2r_init'] = []
        pred['T_q2r_opt'] = []
        pred['T_q2r_opt_list'] = []
        pred['shiftxyr'] = []
        pred['pose_loss'] = []

        with torch.no_grad():
            p3D_ref_init = T_init * p3D_query
            p2D_ref_init, valid_init = data['ref']['camera'].world2image(p3D_ref_init)
            # p2D_ref_init = self.grid_processor.process(p2D_ref_init)
            B, C, A, A = data['ref']['image'].size()
            p2D_ref_init = torch.clamp(p2D_ref_init, min=0., max=A)
            p2D_ref_init_ = p2D_ref_init.long()

            if self.conf.geo_encoder == 'dense':
                p2D_img = torch.zeros((B, 1, A, A), dtype=torch.float32).to(data['ref']['image'].device)
                for b in range(B):
                    p2D_img[b, :, p2D_ref_init_[b, :, 1], p2D_ref_init_[b, :, 0]] = 1.
                p2D_ref_feat = self.geo_encoder(p2D_img, p2D_ref_init)
                if self.conf.debug:
                    p3D_ref_gt = data['T_q2r_gt'] * p3D_query
                    p2D_ref_gt, valid_gt = data['ref']['camera'].world2image(p3D_ref_gt)
                    # p2D_ref_init = self.grid_processor.process(p2D_ref_init)
                    B, C, A, A = data['ref']['image'].size()
                    p2D_ref_gt = torch.clamp(p2D_ref_gt, min=0., max=A)
                    p2D_ref_gt_ = p2D_ref_gt.long()
                    p2D_img_gt = torch.zeros((B, 1, A, A), dtype=torch.float32).to(data['ref']['image'].device)
                    for b in range(B):
                        p2D_img_gt[b, :, p2D_ref_gt_[b, :, 1], p2D_ref_gt_[b, :, 0]] = 1.

                    from pixloc.visualization.viz_2d import imsave
                    imsave(p2D_img[0], '/ws/external/debug_images/geo', 'p2D_img')
                    imsave(p2D_img_gt[0], '/ws/external/debug_images/geo', 'p2D_img_gt')
                    imsave(data['ref']['image'][0], '/ws/external/debug_images/geo', 'ref')
            elif self.conf.geo_encoder == 'sp2d_p':
                B, N, _ = p2D_ref_init.size()
                p2D_point = torch.ones((B, N, 1), dtype=torch.float32).to(p2D_ref_init.device)
                p2D_ref_feat = self.geo_encoder(p2D_point, p2D_ref_init_, spatial_shape=(A, A), batch_size=B)
            elif self.conf.geo_encoder == 'sp2d_pxyz':
                B, N, _ = p2D_ref_init.size()
                p2D_point = torch.ones((B, N, 1), dtype=torch.float32).to(p2D_ref_init.device)
                p3D_ref_init_normalized = (p3D_ref_init - p3D_ref_init.mean(dim=1, keepdim=True)) / (p3D_ref_init.std(dim=1, keepdim=True) + 1e-6)
                p2D_point = torch.cat([p2D_point, p3D_ref_init_normalized], dim=-1)
                p2D_ref_feat = self.geo_encoder(p2D_point, p2D_ref_init_, spatial_shape=(A, A), batch_size=B)
            elif self.conf.geo_encoder == 'sp2d_pz':
                B, N, _ = p2D_ref_init.size()
                p2D_point = torch.ones((B, N, 1), dtype=torch.float32).to(p2D_ref_init.device)
                p3D_ref_init_normalized = (p3D_ref_init - p3D_ref_init.mean(dim=1, keepdim=True)) / (p3D_ref_init.std(dim=1, keepdim=True) + 1e-6)
                p2D_point = torch.cat([p2D_point, p3D_ref_init_normalized[..., -1:]], dim=-1)
                p2D_ref_feat = self.geo_encoder(p2D_point, p2D_ref_init_, spatial_shape=(A, A), batch_size=B)
            elif self.conf.geo_encoder == 'sp2d_pza':
                B, N, _ = p2D_ref_init.size()
                p2D_point = torch.ones((B, N, 1), dtype=torch.float32).to(p2D_ref_init.device)
                p3D_ref_init_normalized = (p3D_ref_init - p3D_ref_init.mean(dim=1, keepdim=True)) / (
                            p3D_ref_init.std(dim=1, keepdim=True) + 1e-6)
                angle = p3D_query[..., 2] / torch.sqrt(p3D_query[..., 0] ** 2 + p3D_query[..., 2] ** 2)
                p2D_point = torch.cat([p2D_point, p3D_ref_init_normalized[..., -1:], angle.unsqueeze(dim=-1)], dim=-1)
                p2D_ref_feat = self.geo_encoder(p2D_point, p2D_ref_init_, spatial_shape=(A, A), batch_size=B)

        if self.conf.debug:
            path = 'debug_images/geo'  # 'visualizations/dense'
            from pixloc.pixlib.geometry.wrappers import project_grd_to_map, project_map_to_grd
            r2q_img, r2q_mask, p3d_grd, _ = project_map_to_grd(data['T_q2r_gt'], data['query']['camera'].cuda(),
                                                               data['ref']['camera'].cuda(),
                                                               data['query']['image'], data['ref']['image'], data)

            q2r_img, q2r_mask, _, _ = project_grd_to_map(data['T_q2r_gt'], data['query']['camera'].cuda(),
                                                         data['ref']['camera'].cuda(),
                                                         data['query']['image'], data['ref']['image'], data)

            from pixloc.visualization.viz_2d import imsave
            imsave(q2r_img[0], f'/ws/external/{path}', '0q2r')
            imsave(data['query']['image'][0], f'/ws/external/{path}', '0grd')
            imsave(data['ref']['image'][0], f'/ws/external/{path}', '0sat')
            imsave(r2q_img[0], f'/ws/external/{path}', '1r2q')
            imsave(data['query']['image'][0], f'/ws/external/{path}', '1grd')

            imsave(data['ref']['image'][0], f'/ws/external/{path}', '1sat')
            # print(f"roll: {data['roll']}, pitch: {data['pitch']}")

            p3D_q = data['query']['points3D']
            p3D_r_gt = data['T_q2r_gt'] * p3D_q
            p3D_r_init = data['T_q2r_init'] * p3D_q

            cam_r = data['ref']['camera']
            p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
            p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
            p2D_r_init, _ = cam_r.world2image(p3D_r_init)

            from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear
            from pixloc.visualize_3dvoxel import project
            F_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_q)
            F_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_gt)

            grd_proj_color = project(F_q, data['ref']['image'], p2D_r_gt)
            sat_color_gt = project(F_r_gt, data['ref']['image'], p2D_r_gt)

            imsave(grd_proj_color.permute(2, 0, 1), f'/ws/external/{path}', '1grd_proj_color_gt')
            imsave(sat_color_gt.permute(2, 0, 1), f'/ws/external/{path}', '1sat_color_gt')

            ## angle ##
            angle_proj = project(angle.unsqueeze(dim=-1).repeat(1, 1, 3), data['ref']['image'], p2D_r_gt)
            imsave(angle_proj.permute(2, 0, 1), f'/ws/external/{path}', '1angle_proj_gt')


        for i in reversed(range(len(self.extractor.scales))):
            if self.conf.optimizer.attention:
                F_ref = pred['ref']['feature_maps'][i] * pred['ref']['confidences'][i]
            else:
                F_ref = pred['ref']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]

            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
                opt.nnrefine.initialize_rsum(i)
            else:
                opt = self.optimizer
                opt.nnrefine.initialize_rsum(i)

            if self.conf.optimizer.attention:
                F_q = pred['query']['feature_maps'][i] * pred['query']['confidences'][i]
            else:
                F_q = pred['query']['feature_maps'][i]
            cam_q = pred['query']['camera_pyr'][i]

            p2D_query, visible = cam_q.world2image(data['query']['T_w2cam']*p3D_query)
            F_q, mask, _ = opt.interpolator(F_q, p2D_query)
            mask &= visible

            if self.conf.optimizer.jacobian:
                W_q = pred['query']['confidences'][i]
                W_q, _, _ = opt.interpolator(W_q, p2D_query)
                W_ref = pred['ref']['confidences'][i]
                W_ref_q = (W_ref, W_q, 1)
            else:
                W_q = pred['query']['confidences'][i]
                W_q, _, _ = opt.interpolator(W_q, p2D_query)
                W_ref = pred['ref']['confidences'][i]
                W_ref_q = (W_ref, W_q, 1)

            if self.conf.normalize_features in ['l2', True]:
                F_q = nnF.normalize(F_q, dim=2)  # B x N x C
                F_ref = nnF.normalize(F_ref, dim=1)  # B x C x W x H
            elif self.conf.normalize_features == 'zsn':
                F_q = (F_q - F_q.mean(dim=2, keepdim=True)) / (F_q.std(dim=2, keepdim=True) + 1e-6)
                F_ref = (F_ref - F_ref.mean(dim=1, keepdim=True)) / (F_ref.std(dim=1, keepdim=True) + 1e-6)


            # if self.conf.optimizer.multi_pose > 1:
            #     B = F_q.size(0)
            #     pose_estimator_input = dict(
            #         p3D=p3D_query, F_ref=F_ref, F_q=F_q, T_init=T_init, camera=cam_ref,
            #         mask=mask, W_ref_q=W_ref_q, data=data, scale=i) # TODO
            #     pose_estimator_input = self.repeat_features(pose_estimator_input, repeat=self.conf.optimizer.multi_pose) # TODO
            #     T_opt, failed = opt(pose_estimator_input)
            #     T_opt = Pose(T_opt._data[:B])  # TODO
            # else:

            T_opt, failed, T_opt_list = opt(dict(
                p3D=p3D_query, F_ref=F_ref, F_q=F_q, p2D_ref_feat=p2D_ref_feat, T_init=T_init, camera=cam_ref,
                mask=mask, W_ref_q=W_ref_q, data=data, scale=i))


            pred['T_q2r_init'].append(T_init)
            pred['T_q2r_opt'].append(T_opt)
            pred['T_q2r_opt_list'].append(T_opt_list)
            # pred['shiftxyr'].append(shiftxyr)

            if self.conf.optimizer.cascade:
                T_init = T_opt
            else:
                T_init = T_opt.detach()     # default

            # query & reprojection GT error, for query unet back propogate  # PAB Loss
            if self.conf.optimizer.pose_loss == 'triplet': #pose_loss:
                loss_gt = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                loss_init = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                diff_loss = self.conf.optimizer.pose_lambda * torch.log(1 + torch.exp(10*(1- (loss_init + 1e-8) / (loss_gt + 1e-8))))
                pred['pose_loss'].append(diff_loss)
            elif self.conf.optimizer.pose_loss == 'triplet2':
                loss_gt = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask,
                                              W_ref_query=W_ref_q)
                loss_init = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref, mask=mask,
                                                W_ref_query=W_ref_q)
                loss_fusion_gt = self.preject_fusion_loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref,
                                                          mask=mask, W_ref_query=W_ref_q)
                loss_fusion_init = self.preject_fusion_loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref,
                                                            mask=mask,
                                                            W_ref_query=W_ref_q)
                diff_loss = self.conf.optimizer.pose_lambda * torch.log(
                    1 + torch.exp(10 * (1 - (loss_init + loss_fusion_init + 1e-8) / (loss_gt + loss_fusion_gt + 1e-8))))
                pred['pose_loss'].append(diff_loss)

            elif self.conf.optimizer.pose_loss == 'triplet3':
                loss_gt = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                loss_init = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                diff_loss1 = self.conf.optimizer.pose_lambda * torch.log(1 + torch.exp(10*(1- (loss_init + 1e-8) / (loss_gt + 1e-8))))

                loss_gt = self.preject_geo_l1loss(opt, p3D_query, F_ref, F_q, p2D_ref_feat, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q, scale=i)
                loss_init = self.preject_geo_l1loss(opt, p3D_query, F_ref, F_q, p2D_ref_feat, data['T_q2r_init'], cam_ref, mask=mask, W_ref_query=W_ref_q, scale=i)
                diff_loss2 = self.conf.optimizer.pose_lambda * torch.log(1 + torch.exp(10 * (1 - (loss_init + 1e-8) / (loss_gt + 1e-8))))

                diff_loss = diff_loss1 + diff_loss2
                pred['pose_loss'].append(diff_loss)

            elif self.conf.optimizer.pose_loss == 'rr':
                diff_loss = 0
                for i, res in enumerate(opt.nnrefine.r_sum[2-i]):
                    if i == 0:
                        res_prev = res
                    else:
                        # diff_loss = (res**2).sum() - (res_prev**2).sum()
                        # pred['pose_loss'].append(diff_loss)
                        err = (res**2).sum(-1) - (res_prev**2).sum(-1)  # [B, N, C] -> [B, N]
                        err = scaled_barron(1., 2.)(err)[0] / 4
                        err = masked_mean(err, mask, -1)    # [B, N] -> [B]
                        err = torch.max(torch.zeros_like(err), err)
                        diff_loss += self.conf.optimizer.pose_lambda * err
                pred['pose_loss'].append(diff_loss)

        return pred

    def repeat_features(self, features, repeat):
        new_features = dict()
        for key, value in features.items():
            if isinstance(value, Camera):
                new_value = Camera(value._data.repeat_interleave(repeat, dim=0))
            elif isinstance(value, Pose):
                new_value = Pose(value._data.repeat_interleave(repeat, dim=0))
            elif isinstance(value, torch.Tensor):
                new_value = value.repeat_interleave(repeat, dim=0)
            elif isinstance(value, tuple):
                new_value = ()
                for v in value:
                    if isinstance(v, torch.Tensor):
                        v = v.repeat_interleave(repeat, dim=0)
                    new_value = new_value + (v,)
            elif isinstance(value, dict):
                new_value = dict()
                new_value['mean'] = value['mean'].repeat_interleave(repeat, dim=0).detach()
                new_value['std'] = value['std'].repeat_interleave(repeat, dim=0).detach()
                new_value['shift_range'] = value['shift_range'].repeat_interleave(repeat, dim=0).detach()
                # new_value = value
            else:
                new_value = value
            new_features[key] = new_value
        return new_features


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

    def preject_geo_l1loss(self, opt, p3D, F_ref, F_query, p2D_ref_feat,
                           T_gt, camera, mask=None, W_ref_query= None, scale=None):
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        p3D_r = T_gt * p3D  # q_3d to q2r_3d
        p2D, visible = camera.world2image(p3D_r)  # q2r_3d to q2r_2d
        ref_feat, valid, gradients = opt.interpolator(F_ref, p2D, return_gradients=False)  # get g2r 2d features

        if 2 - scale == 0:
            geo_proj = opt.nnrefine.geo_linear0(ref_feat)
        elif 2 - scale == 1:
            geo_proj = opt.nnrefine.geo_linear1(ref_feat)
        elif 2 - scale == 2:
            geo_proj = opt.nnrefine.geo_linear2(ref_feat)
        geo_proj = opt.nnrefine.geo_proj(geo_proj)
        p2D_ref_feat = opt.nnrefine.geo_proj(p2D_ref_feat)
        res = geo_proj - p2D_ref_feat

        # res, valid, w_unc, _, _ = opt.cost_fn.residuals(T_gt, *args)
        # if mask is not None:
        #     valid &= mask



        # compute the cost and aggregate the weights
        cost = (res ** 2).sum(-1)
        cost, w_loss, _ = opt.loss_fn(cost) # robust cost
        loss = cost * valid.float()
        # if w_unc is not None:
        #     loss = loss * w_unc

        return torch.sum(loss, dim=-1)/(torch.sum(valid)+1e-6)

    def preject_fusion_loss(self, opt, p3D, F_ref, F_query, T_gt, camera, mask=None, W_ref_query= None):
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        _, valid, w_unc, F_r, _ = opt.cost_fn.residuals(T_gt, *args)
        if mask is not None:
            valid &= mask

        p3D_ref = torch.nn.functional.normalize(T_gt * p3D, dim=-1)
        p3D_ref_feat = torch.nn.functional.normalize(opt.nnrefine.linearp(p3D_ref), dim=-1)
        res = F_r - p3D_ref_feat

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
        elif self.conf.optimizer.main_loss == 'tf':
            losses = self.tf_loss(pred, data)
        elif self.conf.optimizer.main_loss == 'metric':
            losses = self.metric_loss(pred, data)
        elif self.conf.optimizer.main_loss == 'reproj_distance':
            losses = self.reproj_distance_loss(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == "reproj2":
            losses = self.reproj_loss2(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == "reproj2r":
            losses = self.reproj_loss2r(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == "reproj2tf":
            losses = self.reproj_loss2tf(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'reproj3':
            losses = self.reproj_loss3(pred, data)  # default = reproj
        elif self.conf.optimizer.main_loss == 'reproj_mask':
            losses = self.reproj_loss_mask(pred, data)  # default = reproj
        else:
            losses = self.reproj_loss(pred, data)  # default = reproj

        return losses

    def tf_loss(self, pred, data):
        T_q2r_gt = data['T_q2r_gt']
        T_q2r_init = data['T_q2r_init']

        def tf_error(T_q2r):
            err_R = torch.sum(torch.sum(torch.abs(T_q2r_gt.R - T_q2r.R), dim=-1), dim=-1)
            err_T = torch.sum(torch.abs(T_q2r_gt.t - T_q2r.t), dim=-1)
            err = self.conf.optimizer.coe_rot * err_R + self.conf.optimizer.coe_lat * err_T
            return err

        err_init = tf_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss != 'none':
            losses['pose_loss'] = 0
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = tf_error(T_opt)
            loss = err / num_scales
            losses[f'tf_error/{i}'] = err
            losses['total'] += loss

        losses['tf_error'] = err
        losses['tf_error/init'] = err_init

        # with torch.no_grad():
        #     reproj_losses = self.reproj_loss(pred, data)
        #     losses['reprojection_error'] = reproj_losses['reprojection_error'].mean().clone().detach()

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
            loss = (coe * err).sum(dim=-1) / num_scales

            losses[f'shift_error/{i}'] = err.mean(dim=-1).detach()
            # losses[f'error_lat/{i}'] = err[:, 0].detach()
            # losses[f'error_lon/{i}'] = err[:, 1].detach()
            # losses[f'error_rot/{i}'] = err[:, 2].detach()

            losses['total'] += loss

        losses['shift_error'] = err.mean(dim=-1).detach()
        losses['shift_error/init'] = err_init.mean(dim=-1).detach()

        # with torch.no_grad():
        #     reproj_losses = self.reproj_loss(pred, data)
        #     losses['reprojection_error'] = reproj_losses['reprojection_error'].detach()

        return losses

    def reproj_distance_loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']
        distance = torch.sqrt(points_3d.pow(2).mean(dim=-1))
        distance = distance / distance[:, :1]

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = err * distance
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss != 'none':
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
            if self.conf.optimizer.pose_loss != 'none':
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

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
        # success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss != 'none':
            losses['pose_loss'] = 0

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            # if i > 0:
            #     loss = loss * success.float()
            # thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            # success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss != 'none':
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses

    def reproj_loss_mask(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i & data['query']['points3D_mask']).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        # success = None
        losses = {'total': 0.}
        if self.conf.optimizer.pose_loss != 'none':
            losses['pose_loss'] = 0

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            # if i > 0:
            #     loss = loss * success.float()
            # thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            # success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss != 'none':
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

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
        if self.conf.optimizer.pose_loss != 'none':
            losses['pose_loss'] = 0

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # query & reprojection GT error, for query unet back propogate
            if self.conf.optimizer.pose_loss != 'none':
                losses['pose_loss'] += pred['pose_loss'][i] / num_scales
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i] / num_scales).clamp(
                    max=self.conf.clamp_error / num_scales)


        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses

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
            loss = (err[0] + err[1]).mean()
            metrics['total'] += loss / num_scales

            metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[
                f'long_error/{i}'] = err
        metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error'] = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[
            f'long_error/init'] = err_init

        with torch.no_grad():
            reproj_losses = self.reproj_loss(pred, data)
            metrics['reprojection_error'] = reproj_losses['reprojection_error'].detach()

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


    def metrics_analysis(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_q2r):
            # err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
            # err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            return err_R, err_t, err_lat, err_long

        metrics = {}
        # error init
        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[
            f'long_error/init'] = err_init

        # error pred
        pred['T_q2r_opt_list'] = list(itertools.chain(*pred['T_q2r_opt_list']))
        R_error, t_error, lat_error, long_error = (torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device))


        for j, T_opt in enumerate(pred['T_q2r_opt_list']):
            err = scaled_pose_error(T_opt)
            # R_error, t_error, lat_error, lon_error = err
            R_error = torch.cat([R_error, err[0]])
            t_error = torch.cat([t_error, err[1]])
            lat_error = torch.cat([lat_error, err[2]])
            long_error = torch.cat([long_error, err[3]])

        metrics['R_error'] = R_error
        metrics['t_error'] = t_error
        metrics['lat_error'] = lat_error
        metrics['long_error'] = long_error

        return metrics

    # def metrics_analysis(self, pred, data):
    #     T_r2q_gt = data['T_q2r_gt'].inv()
    #
    #     @torch.no_grad()
    #     def scaled_pose_error(T_q2r):
    #         # err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
    #         # err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
    #         err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
    #         err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
    #         return err_R, err_t, err_lat, err_long
    #
    #     metrics = {}
    #     for i, T_opt in enumerate(pred['T_q2r_opt']):
    #         err = scaled_pose_error(T_opt)
    #         metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[f'long_error/{i}'] = err
    #     metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error']  = err
    #
    #     err_init = scaled_pose_error(pred['T_q2r_init'][0])
    #     metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[f'long_error/init'] = err_init
    #
    #     pred['T_q2r_opt_list'] = list(itertools.chain(*pred['T_q2r_opt_list']))
    #
    #     R_error_max, t_error_max = torch.zeros_like(err[0]), torch.zeros_like(err[0])
    #     R_R1, t_R1 = torch.tensor([]).to(err[0].device), torch.tensor([]).to(err[0].device)
    #
    #     for j, T_opt in enumerate(pred['T_q2r_opt_list']):
    #         err = scaled_pose_error(T_opt)
    #         R_error, t_error, lat_error, long_error = err
    #
    #         R_error_max = torch.max(R_error_max, R_error)
    #         t_error_max = torch.max(t_error_max, t_error)
    #
    #         R_R1 = torch.cat([R_R1, (R_error < 1).unsqueeze(dim=1)], dim=1)
    #         t_R1 = torch.cat([t_R1, (t_error < 1).unsqueeze(dim=1)], dim=1)
    #
    #     R_R1_first_index = torch.argmax(R_R1.float(), dim=1)
    #     has_true = torch.any(R_R1, dim=1)
    #     R_R1_first_index[~has_true] = -1
    #
    #     t_R1_first_index = torch.argmax(t_R1.float(), dim=1)
    #     has_true = torch.any(t_R1, dim=1)
    #     t_R1_first_index[~has_true] = -1
    #
    #     metrics['R_error_max'] = R_error_max
    #     metrics['t_error_max'] = t_error_max
    #     metrics['R_min_iter'] = R_R1_first_index
    #     metrics['t_min_iter'] = t_R1_first_index
    #
    #     return metrics
