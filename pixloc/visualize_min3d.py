
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import open3d as o3d

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
    features_to_RGB, add_text, save_plot, plot_valid_points, get_warp_sat2real)

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

    for idx, data in zip(range(2959), val_loader):
        p3D_q = data['query']['points3D']
        p3D_r_gt = data['T_q2r_gt'] * p3D_q
        p3D_r_init = data['T_q2r_init'] * p3D_q

        cam_r = data['ref']['camera']
        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(p3D_r_gt)
        p2D_r_init, _ = cam_r.world2image(p3D_r_init)

        # p2D_q, visible = cam_q.world2image(data['query']['T_w2cam'] * p3D_q)
        from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear
        F_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_q)
        F_r_gt, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_gt)
        F_r_init, _ = interpolate_tensor_bilinear(data['ref']['image'], p2D_r_init)
        F_res_gt = F_q - F_r_gt
        F_res_gt = (F_res_gt - F_res_gt.mean()) / (F_res_gt.std() + 1e-6)
        F_res_init = F_q - F_r_init
        F_res_init = (F_res_init - F_res_init.mean()) / (F_res_init.std() + 1e-6)

        pcd_q = o3d.geometry.PointCloud()
        pcd_q.points = o3d.utility.Vector3dVector(p3D_q[0].cpu().numpy())
        pcd_q.colors = o3d.utility.Vector3dVector(F_q[0].cpu().numpy())

        pcd_r_gt = o3d.geometry.PointCloud()
        pcd_r_gt.points = o3d.utility.Vector3dVector(p3D_r_gt[0].cpu().numpy())
        pcd_r_gt.colors = o3d.utility.Vector3dVector(F_r_gt[0].cpu().numpy())

        pcd_r_init = o3d.geometry.PointCloud()
        pcd_r_init.points = o3d.utility.Vector3dVector(p3D_r_init[0].cpu().numpy())
        pcd_r_init.colors = o3d.utility.Vector3dVector(F_r_init[0].cpu().numpy())

        pcd_res_gt = o3d.geometry.PointCloud()
        pcd_res_gt.points = o3d.utility.Vector3dVector(p3D_r_gt[0].cpu().numpy())
        pcd_res_gt.colors = o3d.utility.Vector3dVector(F_res_gt[0].cpu().numpy())

        pcd_res_init = o3d.geometry.PointCloud()
        pcd_res_init.points = o3d.utility.Vector3dVector(p3D_r_init[0].cpu().numpy())
        pcd_res_init.colors = o3d.utility.Vector3dVector(F_res_init[0].cpu().numpy())

        o3d.io.write_point_cloud(os.path.join(save_path, 'pcd_q.ply'), pcd_q)
        o3d.io.write_point_cloud(os.path.join(save_path, 'pcd_r_gt.ply'), pcd_r_gt)
        o3d.io.write_point_cloud(os.path.join(save_path, 'pcd_r_init.ply'), pcd_r_init)
        o3d.io.write_point_cloud(os.path.join(save_path, 'pcd_res_gt.ply'), pcd_res_gt)
        o3d.io.write_point_cloud(os.path.join(save_path, 'pcd_res_init.ply'), pcd_res_init)

        # o3d.visualization.draw_geometries([pcd_q])
        # o3d.visualization.draw_geometries([pcd_r_gt])
        # o3d.visualization.draw_geometries([pcd_r_init])

    return 0

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
        from pixloc.pixlib.datasets.kitti import Kitti

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

