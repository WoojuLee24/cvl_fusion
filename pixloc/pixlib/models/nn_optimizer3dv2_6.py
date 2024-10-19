import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

# from .pointnet import PointNetEncoder, PointNetEncoder1_1
# from .pointnet2 import PointNetEncoder2
# from .pointnet2_1 import PointNetEncoder2_1
from pixloc.pixlib.models.mlp_mixer import MLPMixer
from pixloc.pixlib.models.simplevit import SimpleViT, Transformer, CrossTransformer
from pixloc.pixlib.geometry.optimization import optimizer_step, optimizer_pstep
from pixloc.pixlib.models.sparse_conv import SparseNet

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
        pose_loss='triplet', # 'triplet', 'rr'
        pose_lambda=1,
        main_loss='reproj',    # reproj
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
        geo_version=1.0,# deprecated
        attention=False,
        mask='visible',
        weights=False,
        input_dim=[128, 128, 32],  # [32, 128, 128],
        normalize_geometry='none',
        normalize_geometry_feature='l2', #'none',
        normalize_J='none',  # l2
        jacobian=True,
        jtr=False,
        integral=False,
        kp=1.,
        kd=1.,
        ki=1.,
        multi_pose=1,
        dropout=0.2,
        max_num_points3D=5000,
        max_num_out_points3D=15000,
        max_num_features=5000,
        voxel_shape=[400, 400, 40],
        max_volume_space=[100, 100, 10],
        min_volume_space=[-100, -100, -5],
        stride=[2, 2, 2],
        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.conf = conf
        num_lambda = 3 if self.conf.pose_from == 'rt' else 6
        self.dampingnet = DampingNet(conf.damping, num_lambda)
        if self.conf.nnrefine == 'v0.1':
            self.nnrefine = NNrefinev0_1(conf)
        elif self.conf.nnrefine == 'v1.0':
            self.nnrefine = NNrefinev1_0(conf)
        elif self.conf.nnrefine == 'v1.1':
            self.nnrefine = NNrefinev1_1(conf)
        elif self.conf.nnrefine == 'v2.0':
            self.nnrefine = NNrefinev2_0(conf)
        assert conf.learned_damping
        super()._init(conf)


    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['p2D_ref_feat'],
            data['T_init'], data['camera'], data['mask'], data.get('W_ref_q'), data, data['scale'])


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor, p2D_ref_feat: Tensor,
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
        args = (self.conf.pose_from, camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()
        # shiftxyr = torch.zeros_like(shift_range)
        T_opt_list = []

        mean = data['data']['mean']
        std = data['data']['std']
        if self.conf.normalize_geometry == 'zsn':  # deprecated
            p3D = (p3D - p3D.mean()) / (p3D.std() + 1e-6)
        elif self.conf.normalize_geometry == 'l2':  # deprecated
            p3D = torch.nn.functional.normalize(p3D, dim=-1)
        elif self.conf.normalize_geometry == 'zsn2':  # deprecated
            p3D = (p3D - mean) / (std + 1e-6)
        elif self.conf.normalize_geometry == 'zsn3':
            p3D = (p3D - p3D.mean(dim=1, keepdim=True)) / (p3D.std(dim=1, keepdim=True) + 1e-6)

        for i in range(self.conf.num_iters):
            res, valid, w_unc, p3D_ref, F_ref2D, J = self.cost_fn.residual_jacobian3(T, *args)

            # if self.conf.normalize_geometry == 'zsn':   # deprecated
            #     p3D_ref = (p3D_ref - p3D_ref.mean()) / (p3D_ref.std() + 1e-6)
            # elif self.conf.normalize_geometry == 'l2':  # deprecated
            #     p3D_ref = torch.nn.functional.normalize(p3D_ref, dim=-1)
            # elif self.conf.normalize_geometry == 'zsn2':    # deprecated
            #     p3D_ref = (p3D_ref - mean) / (std + 1e-6)
            # elif self.conf.normalize_geometry == 'zsn3':
            #     p3D_ref = (p3D_ref - p3D_ref.mean(dim=1, keepdim=True)) / (p3D_ref.std(dim=1, keepdim=True) + 1e-6)


            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # delta = self.nnrefine.debug_forward(data, F_ref, res, F_query, F_ref2D, p3D, p3D_ref, J, w_unc, valid, scale, lambda_, failed)
            delta = self.nnrefine(res, F_query, F_ref2D, p3D, p3D_ref, p2D_ref_feat, J, w_unc, valid, scale)

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
                mul_range = torch.tensor([[self.conf.trans_range, self.conf.trans_range, self.conf.rot_range]], dtype=torch.float32)
                mul_range = mul_range.to(shift_range.device)
                shift_range = shift_range * mul_range
                delta = delta * shift_range.detach()
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

            T_opt_list.append(T)

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta)

            # if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost): # TODO
            #     break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, T_opt_list # , shiftxyr


class NNrefinev1_0(nn.Module):
    def __init__(self, args):
        super(NNrefinev1_0, self).__init__()
        self.args = args
        self.p3d_mean = torch.tensor([[[0.3182,  1.6504, 14.9031]]], dtype=torch.float32).cuda()
        self.p3d_std = torch.tensor([[[9.1397,  0.0000, 10.4613]]], dtype=torch.float32).cuda()

        self.cin = self.args.input_dim  # [128, 128, 32]
        self.cout = 96
        pointc = self.args.input_dim[2]
        self.initialize_rsum()

        # positional embedding
        if self.args.linearp == 'basic':
            self.linearp = nn.Sequential(nn.Linear(3, 16),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(16, pointc),
                                         nn.ReLU(inplace=False),
                                         nn.Linear(pointc, pointc))

        if self.args.integral:
            I_size = self.args.input_dim
            self.cin = [c+I_size[i] for i, c in enumerate(self.cin)]
        if self.args.jacobian:
            J_size = self.args.input_dim
            self.cin = [c+J_size[i]*3 for i, c in enumerate(self.cin)]

        if self.args.version == 0:
            self.cin = [c + 3*self.args.input_dim[2] for c in self.cin]
        elif self.args.version in [1.0, 1.1, 1.2, 1.01]:
            self.geo_linear0 = nn.Sequential(nn.ReLU(inplace=False),
                                             nn.Linear(self.args.input_dim[0], self.args.input_dim[2]))
            self.geo_linear1 = nn.Sequential(nn.ReLU(inplace=False),
                                             nn.Linear(self.args.input_dim[0], self.args.input_dim[2]))
            self.geo_linear2 = nn.Sequential(nn.ReLU(inplace=False),
                                             nn.Linear(self.args.input_dim[0], self.args.input_dim[2]))

            if self.args.version in [1.01]:
                self.geo_proj = nn.Identity()
            else:
                self.geo_proj = nn.Sequential(nn.ReLU(inplace=False),
                                              nn.Linear(self.args.input_dim[2], self.args.input_dim[2]),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(self.args.input_dim[2], self.args.input_dim[2]))
            if self.args.version in [1.0, 1.01]:
                self.cin = [c+4*self.args.input_dim[2] for c in self.cin]
            elif self.args.version == 1.1:
                self.cin = [c+2*self.args.input_dim[2] for c in self.cin]
            elif self.args.version == 1.2:
                self.cin = [c+self.args.input_dim[2] for c in self.cin]


        if self.args.pose_from == 'aa':
            self.yout = 6
        elif self.args.pose_from == 'rt':
            self.yout = 3

        self.linear0 = nn.Sequential(nn.ReLU(inplace=False),
                                     nn.Linear(self.cin[0], self.cout))
        self.linear1 = nn.Sequential(nn.ReLU(inplace=False),
                                     nn.Linear(self.cin[1], self.cout))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=False),
                                     nn.Linear(self.cin[2], self.cout))

        # if self.args.pool == 'none':
        if self.args.net == 'mlp':  # default
            num_points = self.args.max_num_points3D + self.args.max_num_out_points3D
            self.pooling = nn.Sequential(nn.ReLU(inplace=False),
                                         nn.Linear(num_points, 256),
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


    def forward(self, res, query_feat, ref_feat, p3D_query, p3D_ref, p2D_ref_feat,
                J, w_unc, valid, scale):

        B, N, C = query_feat.size()
        self.r_sum[2 - scale].append(res)

        # masking
        if self.args.mask == 'all':
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            res = res * valid
            query_feat = query_feat * valid
            ref_feat = ref_feat * valid
            J = J * valid.unsqueeze(dim=-1).detach()
            w_unc = w_unc * valid
        elif self.args.mask == 'rgb':
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            # res = res * valid
            query_feat = query_feat * valid
            ref_feat = ref_feat * valid
            res = ref_feat - query_feat
            J = J * valid.unsqueeze(dim=-1).detach()
        elif self.args.mask in ['none', 'p3d']:
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            # res = res * valid
            query_feat = query_feat * valid
            res = ref_feat - query_feat
        elif self.args.mask == 'res':
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            query_feat = query_feat * valid
            res = (ref_feat - query_feat) * valid
        elif self.args.mask == 'pe':  # developing TODO
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            query_feat = query_feat * valid
            res = ref_feat - query_feat
            res = torch.cat([res, valid], dim=-1)
            query_feat = torch.cat([query_feat, valid], dim=-1)
            ref_feat = torch.cat([ref_feat, valid], dim=-1)

        if self.args.weights:
            res = res * w_unc
            J = J * w_unc.unsqueeze(dim=-1)

        # point feature [bnc]->[bnc']
        if self.args.normalize_geometry == 'zsn3':
            p3D_ref_ = (p3D_ref - p3D_ref.mean(dim=1, keepdim=True)) / (p3D_ref.std(dim=1, keepdim=True) + 1e-6)
        else:
            p3D_ref_ = p3D_ref
        p3D_query_feat = self.linearp(p3D_query.contiguous())
        p3D_ref_feat = self.linearp(p3D_ref_.contiguous())

        # normalization
        if self.args.normalize_geometry_feature == 'l2':
            p3D_query_feat = torch.nn.functional.normalize(p3D_query_feat, dim=-1)
            p3D_ref_feat = torch.nn.functional.normalize(p3D_ref_feat, dim=-1)

        if self.args.mask == 'all':
            p3D_query_feat = p3D_query_feat * valid
            p3D_ref_feat = p3D_ref_feat * valid
        elif self.args.mask == 'pe':
            p3D_query_feat = torch.cat([p3D_query_feat, valid], dim=-1)
            p3D_ref_feat = torch.cat([p3D_ref_feat, valid], dim=-1)
        elif self.args.mask == 'p3d':
            p3D_ref_feat = p3D_ref_feat * valid

        r = self.args.kp * res
        self.r = res
        self.p3D_ref_feat = p3D_ref_feat

        if self.args.integral:
            # self.r_sum[2-scale] += res
            # r = torch.cat([r, self.args.ki * self.r_sum[2-scale]], dim=-1)
            res_sum = sum(self.r_sum[2-scale]) / len(self.r_sum[2-scale])
            r = torch.cat([r, self.args.ki * res_sum], dim=-1)

        if self.args.jacobian:
            J = J.view(B, N, -1)
            if self.args.normalize_J == 'l2':
                J = torch.nn.functional.normalize(J, dim=-1)
            r = torch.cat([r, self.args.kd * J], dim=-1)


        if self.args.version in [0]:
            # default
            r = torch.cat([r, ref_feat, query_feat, p3D_ref_feat], dim=-1)
        elif self.args.version in [1.0, 1.1, 1.2, 1.01]:
            # geometric vs RGB
            if 2 - scale == 0:
                geo_proj = self.geo_linear0(ref_feat)
            elif 2 - scale == 1:
                geo_proj = self.geo_linear1(ref_feat)
            elif 2 - scale == 2:
                geo_proj = self.geo_linear2(ref_feat)
            geo_proj = self.geo_proj(geo_proj)
            p2D_ref_feat = self.geo_proj(p2D_ref_feat)
            r_geo = geo_proj - p2D_ref_feat

            if self.args.version in [1.0, 1.01]:
                r = torch.cat([r, r_geo, ref_feat, query_feat, p3D_ref_feat], dim=-1)
            elif self.args.version == 1.1:
                r = torch.cat([r, r_geo, ref_feat], dim=-1)
            elif self.args.version == 1.2:
                r = torch.cat([r, r_geo], dim=-1)

        # embedding for pose estimation
        B, N, C = r.shape
        if 2-scale == 0:
            x = self.linear0(r)
        elif 2-scale == 1:
            x = self.linear1(r)
        elif 2-scale == 2:
            x = self.linear2(r)

        # point embedding: [bnc] -> [bn'c]
        # channel embedding: [
        if self.args.net in ['mlp']:
            x = x.contiguous().permute(0, 2, 1).contiguous()
            x = self.pooling(x)
            x = x.view(B, -1)
            y = self.mapping(x)  # [B, 3]

        return y

    def initialize_rsum(self, scale=-1):
        # self.r_sum = {0: 0, 1: 0, 2:0}
        # self.r_sum = {0: [], 1: [], 2: []}
        if scale == -1:
            self.r_sum = {0: [], 1: [], 2: []}
        else:
            self.r_sum[2-scale] = []


    def debug_forward(self, data, F_ref, res, query_feat, ref_feat, p3D_query, p3D_ref, J, w_unc, valid, scale, lambda_, failed, integral=False):

        B, N, C = query_feat.size()
        self.r_sum[2 - scale].append(res)
        from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear
        from pixloc.visualize_3dvoxel import project
        from pixloc.visualization.viz_2d import imsave

        data = data['data']
        F_ref2D = data['ref']['image']
        F_query2D = data['query']['image']

        p3D_q = data['query']['points3D']
        p3D_r_gt = data['T_q2r_gt'] * p3D_q
        p3D_r_init = data['T_q2r_init'] * p3D_q

        cam_r = data['ref']['camera']
        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
        p2D_r_init, _ = cam_r.world2image(p3D_r_init)

        F_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_q)
        F_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_gt)
        F_r_init, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_init)

        grd_proj_color_gt = project(F_q, F_ref2D, p2D_r_gt)
        grd_proj_point_gt = project(255, F_ref2D, p2D_r_gt)

        folder = '3dv2_3'
        imsave(grd_proj_color_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_color_gt')
        imsave(grd_proj_point_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_point_gt')

        grd_proj_color_init = project(F_q, data['ref']['image'], p2D_r_init)
        grd_proj_point_init = project(255, data['ref']['image'], p2D_r_init)

        imsave(grd_proj_color_init.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_color_init')
        imsave(grd_proj_point_init.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_point_init')

        res_color_gt = project(F_r_gt - F_q, F_ref2D.mean(dim=1, keepdim=True), p2D_r_gt)
        res_color_init = project(F_r_init - F_q, F_ref2D.mean(dim=1, keepdim=True), p2D_r_init)

        imsave(res_color_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_color_gt')
        imsave(res_color_init.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_color_init')

        res_color_valid_gt = project((F_r_gt - F_q)*valid.unsqueeze(dim=-1), F_ref2D.mean(dim=1, keepdim=True), p2D_r_gt)
        res_color_valid_init = project((F_r_gt - F_q)*valid.unsqueeze(dim=-1), F_ref2D.mean(dim=1, keepdim=True), p2D_r_init)

        imsave(res_color_valid_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_color_valid_gt')
        imsave(res_color_valid_init.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_color_valid_init')



        grd_proj_feature_gt = project(query_feat.mean(dim=-1, keepdim=True), F_ref2D.mean(dim=1, keepdim=True), p2D_r_gt)
        grd_proj_feature_point_gt = project(255, F_ref2D.mean(dim=1, keepdim=True), p2D_r_gt)

        imsave(grd_proj_feature_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_feature_gt')
        imsave(grd_proj_feature_point_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'grd_proj_feature_point_gt')

        res_feature_gt = project(res.mean(dim=-1, keepdim=True), F_ref2D.mean(dim=1, keepdim=True), p2D_r_gt)
        res_feature_init = project(res.mean(dim=-1, keepdim=True), F_ref2D.mean(dim=1, keepdim=True), p2D_r_init)

        imsave(res_feature_gt.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_feature_gt')
        imsave(res_feature_init.permute(2, 0, 1), f'/ws/external/visualizations/{folder}', 'res_feature_init')



        if self.args.mask == 'visible':
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            res = res * valid
            query_feat = query_feat * valid
            ref_feat = ref_feat * valid
            J = J * valid.unsqueeze(dim=-1).detach()
            w_unc = w_unc * valid
        elif self.args.mask == 'all':
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            # res = res * valid
            query_feat = query_feat * valid
            res = ref_feat - query_feat
        elif self.args.mask == 'all_encoding':  # developing TODO
            valid = valid.float().unsqueeze(dim=-1).detach()
            w_unc = w_unc.float().unsqueeze(dim=-1)
            query_feat = query_feat * valid
            res = ref_feat - query_feat
            res = torch.cat([res, valid], dim=-1)
            query_feat = torch.cat([query_feat, valid], dim=-1)
            ref_feat = torch.cat([ref_feat, valid], dim=-1)
            J = torch.cat([J, valid], dim=-1)

        if self.args.weights:
            res = res * w_unc
            J = J * w_unc.unsqueeze(dim=-1)

    def preprocess_lidar_coordinates_with_offset(self, lidar_coords, voxel_size, spatial_shape):
        """
        Args:
        - lidar_coords: [B, N, 3]
        - voxel_size: 0.2
        - spatial_shape: [D, H, W]

        Returns:
        - indices: [N, 4]
        """

        origin_offset = origin_offset = torch.min(lidar_coords, dim=1)[0]
        B, N, C = lidar_coords.shape
        indices_list = []


        max_bound = torch.max(lidar_coords, dim=1)[0]
        min_bound = torch.min(lidar_coords, dim=1)[0]

        if self.fixed_volume_space:
            max_bound = [100, 100, 10]
            min_bound = [-100, -100, -5]




        for b in range(B):
            voxel_coords = ((lidar_coords[b] - origin_offset[b]) / voxel_size).int()  # voxel 단위로 좌표 변환

            mask = (
                    (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < spatial_shape[0]) &
                    (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < spatial_shape[1]) &
                    (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < spatial_shape[2])
            )
            voxel_coords = voxel_coords[mask]

            batch_indices = torch.full((voxel_coords.shape[0], 1), b, dtype=torch.int)  # 해당 배치 번호로 채움
            indices = torch.cat([batch_indices, voxel_coords], dim=1)  # [N, 4] 형태의 인덱스

            indices_list.append(indices)

        # 모든 배치에 대한 인덱스를 결합합니다.
        all_indices = torch.cat(indices_list, dim=0)

        return all_indices


class GridIndexProcessor:
    def __init__(self, grid_size, max_volume_space=None, min_volume_space=None):
        self.grid_size = grid_size
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def process(self, xyz):
        # xyz: [B, N, 3]
        with torch.no_grad():
            B, N, C = xyz.size()

            max_bound = torch.quantile(xyz, 1.0, dim=1).unsqueeze(dim=1)  # 100th percentile equivalent
            min_bound = torch.quantile(xyz, 0.0, dim=1).unsqueeze(dim=1)   # 0th percentile equivalent

            if self.max_volume_space is not None:
                max_bound = torch.tensor(self.max_volume_space)[None, None, :].to(xyz.device)
                min_bound = torch.tensor(self.min_volume_space)[None, None, :].to(xyz.device)

            crop_range = max_bound - min_bound
            cur_grid_size = torch.tensor(self.grid_size).to(crop_range.device)
            cur_grid_size = cur_grid_size[None, None, :]

            intervals = crop_range / (cur_grid_size - 1)
            if (intervals == 0).any():
                print("Zero interval detected!")

            clipped_xyz = torch.clamp(xyz, min=min_bound, max=max_bound)  # xyz 값을 min_bound와 max_bound 사이로 클리핑
            grid_ind = torch.floor((clipped_xyz - min_bound) / intervals).to(torch.int32)  # 그리드 인덱스로 변환

        return grid_ind
