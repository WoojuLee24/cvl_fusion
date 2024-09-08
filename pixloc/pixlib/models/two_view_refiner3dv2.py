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
from pixloc.visualization.viz_2d import features_to_RGB,plot_images,plot_keypoints
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

class TwoViewRefiner3D(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'unet', #'s2dnet',
        },
        'optimizer': {
            'name': 'nn_optimizer3dv2', # 'learned_optimizer', #'basic_optimizer',
            'input': 'res',
            'pose_loss': False,
            'main_loss': 'reproj',
            'coe_lat': 1.,
            'coe_lon': 1.,
            'coe_rot': 1.,
            'cascade': False,
            'attention': False,
            'opt_list': False,
            'jacobian': False,
            'multi_pose': 1,
            'topk': 1,
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

        p3D_query = data['query']['points3D']
        T_init = data['T_q2r_init']
        pred['T_q2r_init'] = []
        pred['T_q2r_opt'] = []
        pred['shiftxyr'] = []
        pred['pose_loss'] = []
        if self.conf.optimizer.multi_pose > 1:
            pred['topk_err'] = 0

        for i in reversed(range(len(self.extractor.scales))):
            if self.conf.optimizer.attention:
                F_ref = pred['ref']['feature_maps'][i] * pred['ref']['confidences'][i]
            else:
                F_ref = pred['ref']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]

            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
                opt.nnrefine.initialize_rsum()   # for integral module
            else:
                opt = self.optimizer
                opt.nnrefine.initialize_rsum()   # for integral module

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
                W_ref_q = None

            if self.conf.normalize_features in ['l2', True]:
                F_q = nnF.normalize(F_q, dim=2)  # B x N x C
                F_ref = nnF.normalize(F_ref, dim=1)  # B x C x W x H
            elif self.conf.normalize_features == 'zsn':
                F_q = (F_q - F_q.mean(dim=2, keepdim=True)) / (F_q.std(dim=2, keepdim=True) + 1e-6)
                F_ref = (F_ref - F_ref.mean(dim=1, keepdim=True)) / (F_ref.std(dim=1, keepdim=True) + 1e-6)

            if self.conf.optimizer.multi_pose > 1:
                B = F_q.size(0)
                pose_estimator_input = dict(
                    p3D=p3D_query, F_ref=F_ref, F_q=F_q, T_init=T_init, camera=cam_ref,
                    mask=mask, W_ref_q=W_ref_q, data=data, scale=i) # TODO
                pose_estimator_input = self.repeat_features(pose_estimator_input,
                                                            repeat=self.conf.optimizer.multi_pose,
                                                            pose=T_init,
                                                            scale=i) # TODO
                T_opt, failed = opt(pose_estimator_input)
                T_opt, topk_err = self.get_topk_err(T_opt, data['T_q2r_gt'], opt.nnrefine.r,
                                             repeat=self.conf.optimizer.multi_pose,
                                             batch_size=B)

                pred['topk_err'] += topk_err

                # T_opt = Pose(T_opt._data[:B])  # TODO
            else:
                T_opt, failed = opt(dict(
                    p3D=p3D_query, F_ref=F_ref, F_q=F_q, T_init=T_init, camera=cam_ref,
                    mask=mask, W_ref_q=W_ref_q, data=data, scale=i))


            pred['T_q2r_init'].append(T_init)
            pred['T_q2r_opt'].append(T_opt)
            # pred['shiftxyr'].append(shiftxyr)

            if self.conf.optimizer.opt_list:
                if self.conf.optimizer.cascade:
                    T_init = T_opt[-1]
                else:
                    T_init = T_opt[-1].detach()
            else:
                if self.conf.optimizer.cascade:
                    T_init = T_opt
                else:
                    T_init = T_opt.detach()     # default

            # query & reprojection GT error, for query unet back propogate  # PAB Loss
            if self.conf.optimizer.pose_loss: #pose_loss:
                loss_gt = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                loss_init = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                diff_loss = torch.log(1 + torch.exp(10*(1- (loss_init + 1e-8) / (loss_gt + 1e-8))))
                pred['pose_loss'].append(diff_loss)

        return pred

    def repeat_features(self, features, repeat, pose, scale):
        new_features = dict()
        B = features['p3D'].size(0)
        assert B==1, "only batch size 1 is tested"
        for key, value in features.items():
            if isinstance(value, Camera):
                new_value = Camera(value._data.repeat_interleave(repeat, dim=0))
            elif isinstance(value, Pose):
                # ramdom shift translation and ratation on yaw
                batch_pose_size = value._data.size(0) // B
                R_yaw, T = self.get_noise(repeat, B, mode='random', scale=scale)
                shifts = Pose.from_Rt(R_yaw, T).cuda().detach()
                new_value = Pose(value._data.repeat_interleave(repeat // batch_pose_size, dim=0))
                new_value = shifts @ new_value
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

    def get_noise(self, repeat, batch_size, mode='random', scale=2):

        rot_range = 0 # 15 / 2 / (3-scale)
        trans_range = 5 / 2

        # yaw shift
        YawShiftRange = rot_range * torch.pi / 180
        if mode == 'random':
            yaw = (2 * YawShiftRange * torch.rand(repeat) - YawShiftRange)

        yaw[0] = 0
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        R_yaw = torch.zeros((repeat, 3, 3))
        R_yaw[:, 0, 0] = cos_yaw
        R_yaw[:, 0, 1] = -sin_yaw
        R_yaw[:, 1, 0] = sin_yaw
        R_yaw[:, 1, 1] = cos_yaw
        R_yaw[:, 2, 2] = 1
        R_yaw = R_yaw.repeat((batch_size, 1, 1))

        # t shift
        TShiftRange = trans_range
        if mode == 'random':
            T = 2 * TShiftRange * torch.rand(repeat, 3) - TShiftRange
        T[0, :] = 0
        T[:, 2] = 0
        T = T.repeat((batch_size, 1))

        return R_yaw, T

    def get_topk_err(self, T_q2r, T_q2r_gt, F_residual, batch_size, repeat, topk=3):
        with torch.no_grad():
            residual_sum = torch.abs(F_residual.sum(dim=1).sum(dim=1))
            residual_sum = residual_sum.reshape(batch_size, -1)
            _, ind = torch.topk(residual_sum, topk, dim=1, largest=False)
            T_r2q_gt = T_q2r_gt.inv()
            T_r2q_gt = Pose(T_r2q_gt._data.repeat_interleave(repeat, dim=0))

            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            # err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            _, ind_R = torch.topk(err_R.reshape(batch_size, -1), topk, dim=1, largest=False)
            _, ind_T = torch.topk(err_t.reshape(batch_size, -1), topk, dim=1, largest=False)

            T_1_err = (ind[0, 0] == ind_T[0, 0]).float().sum() / 3
            T_topk_err = torch.isin(ind[0], ind_T[0]).float().sum() / topk / 3
            R_1_err = (ind[0, 0] == ind_R[0, 0]).float().sum() / batch_size / 3
            R_topk_err = torch.isin(ind[0], ind_R[0]).float().sum() / topk / 3

            err = torch.tensor([T_1_err, T_topk_err, R_1_err, R_topk_err]).cuda()
        return T_q2r[ind[0]], err

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
        if self.conf.optimizer.pose_loss:
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
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0

        if self.conf.optimizer.opt_list:
            pred['T_q2r_opt'] = list(itertools.chain(*pred['T_q2r_opt']))
            num_scales *= self.conf.optimizer.num_iters

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
            if self.conf.optimizer.pose_loss:
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
        if self.conf.optimizer.pose_loss:
            losses['pose_loss'] = 0

        if self.conf.optimizer.opt_list:
            pred['T_q2r_opt'] = list(itertools.chain(*pred['T_q2r_opt']))
            num_scales *= self.conf.optimizer.num_iters

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
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


    def reproj_loss2r(self, pred, data):
        # cam_ref = data['ref']['camera']
        # points_3d = data['query']['points3D']

        # def project(T_q2r):
        #     return cam_ref.world2image(T_q2r * points_3d)
        #
        # p2D_r_gt, mask = project(data['T_q2r_gt'])
        # p2D_r_i, mask_i = project(data['T_q2r_init'])
        # mask = (mask & mask_i).float()
        #
        # def reprojection_error(T_q2r):
        #     p2D_r, _ = project(T_q2r)
        #     err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
        #     # err = scaled_barron(1., 2.)(err)[0] / 4
        #     err = masked_mean(err, mask, -1)
        #     return err
        #
        # err_init = reprojection_error(pred['T_q2r_init'][0])
        #
        # num_scales = len(self.extractor.scales)
        # losses = {'total': 0.}
        #
        # if self.conf.optimizer.opt_list:
        #     pred['T_q2r_opt'] = list(itertools.chain(*pred['T_q2r_opt']))
        #     num_scales *= self.conf.optimizer.num_iters
        #
        # for i, T_opt in enumerate(pred['T_q2r_opt']):
        #     err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
        #     loss = err / num_scales
        #     losses[f'reprojection_error/{i}'] = err
        #     losses['total'] += loss
        #
        # losses['reprojection_error'] = err
        # losses['reprojection_error/init'] = err_init
        losses1 = self.reproj_loss2(pred, data)
        losses2 = self.rt_loss(pred, data)

        losses = losses1
        for k, v in losses2.items():
            if k == 'total':
                losses[k] = losses[k] + losses2[k]
            else:
                losses[k] = losses2[k]

        return losses


    def reproj_loss2tf(self, pred, data):
        # cam_ref = data['ref']['camera']
        # points_3d = data['query']['points3D']

        # def project(T_q2r):
        #     return cam_ref.world2image(T_q2r * points_3d)
        #
        # p2D_r_gt, mask = project(data['T_q2r_gt'])
        # p2D_r_i, mask_i = project(data['T_q2r_init'])
        # mask = (mask & mask_i).float()
        #
        # def reprojection_error(T_q2r):
        #     p2D_r, _ = project(T_q2r)
        #     err = torch.sum((p2D_r_gt - p2D_r) ** 2, dim=-1)
        #     # err = scaled_barron(1., 2.)(err)[0] / 4
        #     err = masked_mean(err, mask, -1)
        #     return err
        #
        # err_init = reprojection_error(pred['T_q2r_init'][0])
        #
        # num_scales = len(self.extractor.scales)
        # losses = {'total': 0.}
        #
        # if self.conf.optimizer.opt_list:
        #     pred['T_q2r_opt'] = list(itertools.chain(*pred['T_q2r_opt']))
        #     num_scales *= self.conf.optimizer.num_iters
        #
        # for i, T_opt in enumerate(pred['T_q2r_opt']):
        #     err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
        #     loss = err / num_scales
        #     losses[f'reprojection_error/{i}'] = err
        #     losses['total'] += loss
        #
        # losses['reprojection_error'] = err
        # losses['reprojection_error/init'] = err_init
        losses1 = self.reproj_loss2(pred, data)
        losses2 = self.tf_loss(pred, data)

        losses = losses1
        for k, v in losses2.items():
            if k == 'total':
                losses[k] = losses[k] + losses2[k]
            else:
                losses[k] = losses2[k]

        return losses


    def reproj_loss3(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = data['query']['points3D']

        if self.conf.optimizer.normalize_geometry == 'zsn2':
            mean = torch.tensor([-0.1917, 0.9250, 15.6600]).to(points_3d.device).repeat(1, 1, 1)
            std = torch.tensor([6.9589, 0.8642, 11.5166]).to(points_3d.device).repeat(1, 1, 1)
            points_3d = (points_3d - mean) / (std + 1e-6)

        p3D_r_gt =  data['T_q2r_gt'] * points_3d
        p3D_r_i = data['T_q2r_init'] * points_3d

        def reprojection_error(T_q2r):
            p3D_r = T_q2r * points_3d
            err = torch.sum((p3D_r_gt - p3D_r) ** 2, dim=-1)
            return err.mean(dim=-1)

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        losses = {'total': 0.}

        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt)
            loss = err / num_scales
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

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
            if self.conf.optimizer.opt_list:
                T_opt = T_opt[-1]
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[f'long_error/{i}'] = err
        metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error']  = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[f'long_error/init'] = err_init

        return metrics
