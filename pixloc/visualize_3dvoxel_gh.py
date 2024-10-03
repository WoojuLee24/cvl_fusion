
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from jedi.inference import follow_error_node_imports_if_possible
from omegaconf import OmegaConf
import os
from tqdm import tqdm
# import open3d as o3d
import pdb

SavePlt = True
visual_path = 'visual_kitti'

Ford_dataset = False
exp = 'kitti'
import collections
from pixloc.pixlib.utils.tensor import batch_to_device, map_tensor
from pixloc.pixlib.utils.tools import set_seed
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, plot_matches, cm_RdGn,
    features_to_RGB, add_text, save_plot, plot_valid_points, get_warp_sat2real, imsave)

class Logger:
    def __init__(self, optimizers=None):
        self.costs = []
        self.dt = []
        self.t = []
        self.camera_trajectory = []
        self.yaw_trajectory = []
        self.pre_q2r = None

        if optimizers is not None:
            if isinstance(optimizers, collections.Iterable):
                for opt in optimizers:
                    opt.logging_fn = self.log
            else:
                optimizers.logging_fn = self.log

    def log(self, **args):
        if args['i'] == 0:
            self.costs.append([])
            # add init and gt camera
            camera_3D = torch.zeros(1,3).to(args['T_delta'].device)
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_gt'] * camera_3D)
            self.camera_gt = camera_2D[0].cpu().numpy()
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_init'] * camera_3D)
            self.camera_trajectory.append((camera_2D[0].cpu().numpy(), valid[0].cpu().numpy()))
            camera_3D[:, -1] = 2
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_gt'] * camera_3D)
            self.camera_gt_yaw = camera_2D[0].cpu().numpy()
            camera_yaw, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_init'] * camera_3D)
            self.yaw_trajectory.append((camera_yaw[0].cpu().numpy(), valid[0].cpu().numpy()))
        # self.costs[-1].append(args['cost'].mean(-1).cpu().numpy())
        self.dt.append(args['T_delta'].magnitude()[1].cpu().numpy())
        self.t.append(args['T'].cpu())

        camera_3D = torch.zeros(1,3).to(args['T_delta'].device)
        camera_2D, valid = self.data['ref']['camera'].world2image(args['T'] * camera_3D)
        camera_3D[:, -1] = 2
        camera_yaw, valid = self.data['ref']['camera'].world2image(args['T'] * camera_3D)
        self.camera_trajectory.append((camera_2D[0].cpu().numpy(), valid[0].cpu().numpy()))
        self.yaw_trajectory.append((camera_yaw[0].cpu().numpy(), valid[0].cpu().numpy()))

        self.pre_q2r = args['T'].cpu()

    def clear_trajectory(self):
        self.camera_trajectory = []
        self.yaw_trajectory = []
        self.t = []

    def set(self, data):
        self.data = data


def min_max_norm(confidence):
    max= torch.max(confidence)
    min= torch.min(confidence)
    normed = (confidence - min) / (max - min + 1e-8)
    return normed

#val
def Val(val_loader, save_path, best_result):
    # refiner.eval()
    from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear
    from pixloc.pixlib.geometry.wrappers import camera_to_onground
    import cv2

    folder_path = '/ws/external/visualizations/3dvoxel/'
    for idx, data in zip(range(2959), val_loader):
        p3D_q = data['query']['points3D']
        p3D_r_gt = data['T_q2r_gt'] * p3D_q
        p3D_r_init = data['T_q2r_init'] * p3D_q

        cam_q = data['query']['camera']
        cam_r = data['ref']['camera']
        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
        p2D_r_init, _ = cam_r.world2image(p3D_r_init)

        # grd lidar projection
        color_image0 = data['query']['image'].cpu().detach().numpy()[0].transpose((1, 2, 0)).copy()
        color_image0 = (color_image0 * 255).astype(np.uint8)
        h, w, c = color_image0.shape
        grd_2d, _ = data['query']['camera'].world2image(data['query']['points3D'])  ##camera 3d to 2d
        grd_2d = grd_2d[0].T.cpu().detach().numpy()
        for j in range(grd_2d.shape[1]):
            x = np.int32(grd_2d[0][j])
            y = np.int32(grd_2d[1][j])
            cv2.circle(color_image0, (x, y), 1, (0, 255, 0), -1)
        plt.figure(figsize=plt.figaspect(0.5))
        plt.imshow(color_image0)
        plt.savefig(os.path.join(folder_path, 'grd_lidar_proj'))
        plt.show()

        # ground homography
        b, c, h, w = data['query']['image'].size()
        device = p3D_q.device
        vv, uu = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        uv = torch.stack([uu, vv], dim=-1)
        uv = uv[None, :, :, :].repeat(b, 1, 1, 1)  # shape = [b, h, w, 2]

        p3D_c = cam_q.image2world(uv)  # [b, h, w, 3]
        p3D_c[..., -1] = torch.ones_like(p3D_c[..., -1])
        p3D_grd = camera_to_onground(p3D_c, data['query']['T_w2cam'], data['query']['camera_h'], data['normal'],
                                     max=100.)
        p3D_grd_r_gt = data['T_q2r_gt'] * p3D_grd.reshape(b, h*w, -1)
        p2D_grd_r_gt, valid_grd_r = cam_r.world2image(p3D_grd_r_gt)
        F_gh_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_grd_r_gt)
        F_gh_q, _ = interpolate_tensor_bilinear(data['query']['image'], uv.reshape(b, h * w, -1))

        gh_proj_color = project(F_gh_q, data['ref']['image'], p2D_grd_r_gt)
        gh_proj_point = project(255, data['ref']['image'], p2D_grd_r_gt)

        imsave(data['ref']['image'][0], '/ws/external/visualizations/3dvoxel/', '0gh_sat')
        imsave(gh_proj_color.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', '0gh_proj_color')
        imsave(gh_proj_point.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', '0gh_proj_point')

        # p2D_q, visible = cam_q.world2image(data['query']['T_w2cam'] * p3D_q)
        F_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_q)
        F_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_gt)
        F_r_init, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_init)
        F_res_gt = F_q - F_r_gt
        F_res_gt = (F_res_gt - F_res_gt.mean()) / (F_res_gt.std() + 1e-6)
        F_res_init = F_q - F_r_init
        F_res_init = (F_res_init - F_res_init.mean()) / (F_res_init.std() + 1e-6)


        # all gh grid
        b, c, h, w = data['query']['image'].size()
        x = torch.linspace(-50, 50, 200, dtype=torch.float32, device=device)
        y = 1.65
        z = torch.linspace(-50, 50, 200, dtype=torch.float32, device=device)
        zz, xx = torch.meshgrid(z, x, indexing='ij')
        p3D_grd_grid = torch.stack([xx, torch.ones_like(xx) * y, zz], dim=-1)
        p3D_grd_grid = p3D_grd_grid.unsqueeze(0).repeat(b, 1, 1, 1)

        p3D_grd_grid_r_gt = data['T_q2r_gt'] * p3D_grd_grid.reshape(b, -1, 3)
        p2D_grd_grid_r_gt, valid_grd_range_r = cam_r.world2image(p3D_grd_grid_r_gt)
        p2D_grd_grid_q, _ = cam_q.world2image(p3D_grd_grid.reshape(b, -1, 3))

        F_grd_grid_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_grd_grid_r_gt)
        F_grd_grid_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_grd_grid_q)

        F_grd_grid_r_gt_color = project(F_grd_grid_r_gt, data['ref']['image'], p2D_grd_grid_r_gt)
        F_grd_grid_r_gt_point = project(255, data['ref']['image'], p2D_grd_grid_r_gt)
        F_grd_grid_q_gt_color = project(F_grd_grid_q, data['ref']['image'], p2D_grd_grid_r_gt)

        imsave(data['ref']['image'][0], '/ws/external/visualizations/3dvoxel/', '1gh_sat')
        imsave(F_grd_grid_r_gt_color.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', '1gh_grd_grid_r_gt_color')
        imsave(F_grd_grid_r_gt_point.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', '1gh_grd_grid_r_gt_point')
        imsave(F_grd_grid_q_gt_color.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', '1gh_grd_grid_q_gt_point')

        # B, N, C = F_q.size()
        # _, _, H, W = data['ref']['image'].size()

        grd_proj_color = project(F_q, data['ref']['image'], p2D_r_gt)
        grd_proj_point = project(255, data['ref']['image'], p2D_r_gt)

        imsave(grd_proj_color.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'grd_proj_color_gt')
        imsave(grd_proj_point.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'grd_proj_point_gt')

        grd_proj_color_init = project(F_q, data['ref']['image'], p2D_r_init)
        grd_proj_point_init = project(255, data['ref']['image'], p2D_r_init)

        imsave(grd_proj_color_init.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'grd_proj_color_init')
        imsave(grd_proj_point_init.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'grd_proj_point_init')

        sat_color_init = project(F_r_init, data['ref']['image'], p2D_r_init)
        imsave(sat_color_init.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'sat_color_init')
        sat_color_gt = project(F_r_gt, data['ref']['image'], p2D_r_gt)
        imsave(sat_color_gt.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'sat_color_gt')


        # p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
        # grd_proj = torch.zeros((H*W, 3), device=p3D_q.device)
        # xys =  p2D_r_gt[0].long() # p2D_r_gt[0].to(torch.int32) # (N, 2)
        # colors = F_q[0] # (N, 3)
        # # (x,y,c) = color
        # indices = xys[:, 1] * W + xys[:, 0]
        # indices = indices.view(-1, 1).expand(-1, 3)
        #
        # grd_proj.scatter_(0, indices.long(), 255)
        # grd_proj = grd_proj.reshape(H, W, C)
        # imsave(grd_proj.permute(2, 0, 1), '/ws/external/visualizations/3dvoxel', 'grd_proj_point')
        #
        # p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
        # grd_proj = torch.zeros((H * W, 3), device=p3D_q.device)
        # xys = p2D_r_gt[0].long()  # p2D_r_gt[0].to(torch.int32) # (N, 2)
        # colors = F_q[0]  # (N, 3)
        # # (x,y,c) = color
        # indices = xys[:, 1] * W + xys[:, 0]
        # indices = indices.view(-1, 1).expand(-1, 3)
        #
        # grd_proj.scatter_(0, indices.long(), colors)
        # grd_proj = grd_proj.reshape(H, W, C)
        # imsave(grd_proj.permute(2,0,1), '/ws/external/visualizations/3dvoxel', 'grd_proj_color')

    return 0

def project(F_q, F_ref2D, p2D_r):
    B, C, H, W = F_ref2D.size()

    grd_proj = torch.ones((H*W, C), device=F_ref2D.device)
    xys = p2D_r[0].long()  # p2D_r_gt[0].to(torch.int32) # (N, 2)
    if isinstance(F_q, torch.Tensor):
        colors = F_q[0]  # (N, 3)
    else:
        colors = 255
    indices = xys[:, 1] * W + xys[:, 0]
    indices = indices.view(-1, 1).expand(-1, C)

    grd_proj.scatter_(0, indices.long(), colors)
    grd_proj = grd_proj.reshape(H, W, C)

    return grd_proj




def test(refiner, test_loader):
    refiner.eval()
    errR = torch.tensor([])
    errlong = torch.tensor([])
    errlat = torch.tensor([])
    for idx, data in enumerate(tqdm(test_loader)):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        metrics = refiner.metrics(pred_, data_)

        errR = torch.cat([errR, metrics['R_error'].cpu().data], dim=0)
        errlong = torch.cat([errlong, metrics['long_error'].cpu().data], dim=0)
        errlat = torch.cat([errlat, metrics['lat_error'].cpu().data], dim=0)

        del pred_, data_

    print(f'acc of lat<=0.25:{torch.sum(errlat <= 0.25) / errlat.size(0)}')
    print(f'acc of lat<=0.5:{torch.sum(errlat <= 0.5) / errlat.size(0)}')
    print(f'acc of lat<=1:{torch.sum(errlat <= 1) / errlat.size(0)}')
    print(f'acc of lat<=2:{torch.sum(errlat <= 2) / errlat.size(0)}')

    print(f'acc of long<=0.25:{torch.sum(errlong <= 0.25) / errlong.size(0)}')
    print(f'acc of long<=0.5:{torch.sum(errlong <= 0.5) / errlong.size(0)}')
    print(f'acc of long<=1:{torch.sum(errlong <= 1) / errlong.size(0)}')
    print(f'acc of long<=2:{torch.sum(errlong <= 2) / errlong.size(0)}')

    print(f'acc of R<=1:{torch.sum(errR <= 1) / errR.size(0)}')
    print(f'acc of R<=2:{torch.sum(errR <= 2) / errR.size(0)}')
    print(f'acc of R<=4:{torch.sum(errR <= 4) / errR.size(0)}')

    print(f'mean errR:{torch.mean(errR)},errlat:{torch.mean(errlat)},errlong:{torch.mean(errlong)}')
    print(f'var errR:{torch.var(errR)},errlat:{torch.var(errlat)},errlong:{torch.var(errlong)}')
    print(f'median errR:{torch.median(errR)},errlat:{torch.median(errlat)},errlong:{torch.median(errlong)}')
    return

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    data_conf = {
        'max_num_points3D': 77777,  # 5000, #both:3976,3D:5000
        'max_num_out_points3D': 777777,
        'force_num_points3D': False,
        'train_batch_size': 1,
        'test_batch_size': 1,
        'num_workers': 1,
        'satmap_zoom': 18,
        'trans_range': 20,
         'rot_range': 15
        # "sampling": 'distance', #'random' #

    }

    if Ford_dataset:
        from pixloc.pixlib.datasets.ford import FordAV

        dataset = FordAV(data_conf)
    else:
        # from pixloc.pixlib.datasets.kitti import Kitti
        from pixloc.pixlib.datasets.kitti3_1 import Kitti

        dataset = Kitti(data_conf)

    torch.set_grad_enabled(False);
    mpl.rcParams['image.interpolation'] = 'bilinear'

    val_loader = dataset.get_data_loader('val', shuffle=True)  # or 'train' ‘val’
    test_loader = dataset.get_data_loader('test', shuffle=False)  # shuffle=True)

    # Name of the example experiment. Replace with your own training experiment.
    device = 'cuda'
    conf = {
        'normalize_dt': False,
        'optimizer': {'num_iters': 1, },
    }
    # refiner = load_experiment(exp, conf, get_last=True).to(device)
    ckpt = '/ws/external/outputs/training/NN3d/baseline_geometry.zsn2.l2_resconcat_reproj2_jac_iters5_c96/checkpoint_best.tar'

    # refiner = load_experiment(exp, conf,
    #                           ckpt=ckpt
    #                           ).to(device)
    save_path = '/ws/external/visualizations/3dcolor' # '/ws/external/checkpoints/Models/3d_res_embed_aap2_iters5_range.False_dup.False/visualizations'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # print(OmegaConf.to_yaml(refiner.conf))

    # logger = Logger(refiner.optimizer)
    # trainning
    set_seed(20)

    if 0: # test
        test(refiner, test_loader) #val_loader
    if 1: # visualization
        # Val(refiner, val_loader, save_path, 0)
        Val(val_loader, save_path, 0)

