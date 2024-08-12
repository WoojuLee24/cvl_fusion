
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from omegaconf import OmegaConf
import os
from tqdm import tqdm

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
def Val(refiner, val_loader, save_path, best_result):
    refiner.eval()
    acc = 0
    cnt = 0
    # only severity 3 is supported
    for idx, data in zip(range(605), val_loader):
        if isinstance(data['query']['image'], dict):
            data['query']['image'] = data['query']['image'][3]
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        cam_r = data['ref']['camera']
        p3D_q = data['query']['points3D']

        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam']*p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        p2D_r_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        p2D_r_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)
        valid = valid_q & valid_r

        losses = refiner.loss(pred_, data_)
        mets = refiner.metrics(pred_, data_)
        errP = f"ΔP {losses['reprojection_error/init'].item():.2f} -> {losses['reprojection_error'].item():.3f} px; "
        errR = f"ΔR {mets['R_error/init'].item():.2f} -> {mets['R_error'].item():.3f} deg; "
        errt = f"Δt {mets['t_error/init'].item():.2f} -> {mets['t_error'].item():.3f} m"
        errlat = f"Δlat {mets['lat_error/init'].item():.2f} -> {mets['lat_error'].item():.3f} m"
        errlong = f"Δlong {mets['long_error/init'].item():.2f} -> {mets['long_error'].item():.3f} m"
        print(errP, errR, errt, errlat,errlong)

        if mets['t_error'].item() < 1 and mets['R_error'].item() < 2:
            acc += 1
        cnt += 1

        # for debug
        if 1:
            imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)
            colors = mpl.cm.cool(1 - np.linspace(0, 1, len(logger.camera_trajectory)))[:, :3]
            plot_images([imr])
            axes = plt.gcf().axes
            for i, (p2s, _), (p2e, _), T, c in zip(range(len(logger.camera_trajectory)), logger.camera_trajectory, logger.yaw_trajectory, logger.t, colors):
                # plot the direction of the body frame
                if i == 0:
                    start_0 = p2s
                    end_0 = p2e
                    axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color='r')
                # elif i == len(logger.camera_trajectory)-1:
                elif i == conf['optimizer']['num_iters'] * 3 - 1:
                    axes[0].quiver(p2s[:, 0], p2s[:, 1], p2e[:, 0] - p2s[:, 0], p2s[:, 1] - p2e[:, 1], color='b')
                    # axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color=c[None])
                # else:
                #     axes[0].quiver(p2s[:,0], p2s[:,1], p2e[:,0]-p2s[:,0], p2s[:,1]-p2e[:,1], color=c[None])
            axes[0].quiver(logger.camera_gt[:, 0], logger.camera_gt[:, 1], logger.camera_gt_yaw[:, 0]-logger.camera_gt[:, 0],
                           logger.camera_gt[:, 1]-logger.camera_gt_yaw[:, 1], color='lime', width=0.005)
            logger.clear_trajectory()
            if SavePlt:
                pose_path = os.path.join(save_path, 'pose')
                if not os.path.exists(pose_path):
                    os.makedirs(pose_path, exist_ok=True)
                save_plot(pose_path+f'/pose_refine{idx:06d}.png')
            plt.show()

    acc = acc/cnt
    print('acc of a epoch:#####',acc)
    # if acc > best_result:
    #     print('best acc:@@@@@', acc)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(refiner.state_dict(), save_path + 'Model_best.pth')
    return acc


def Load_raw(refiner, val_loader, save_path, best_result):
    refiner.eval()
    acc = 0
    cnt = 0
    for idx, data in zip(range(605), val_loader):
        if isinstance(data['query']['image'], dict):
            data['query']['image'] = data['query']['image'][3]
        data_ = batch_to_device(data, device)
        logger.set(data_)
        # pred_ = refiner(data_)
        # pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        cam_r = data['ref']['camera']
        p3D_q = data['query']['points3D']

        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
        # p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        # p2D_r_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        # p2D_r_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)
        # valid = valid_q & valid_r

        # for debug
        if 1:
            imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)
            plot_images([imq], dpi=100)
            if SavePlt:
                imq_path = os.path.join(save_path, 'imq')
                if not os.path.exists(imq_path):
                    os.makedirs(imq_path, exist_ok=True)
                save_plot(imq_path + f'/imq{idx:06d}.png')
            plt.show()

            plot_images([imq], dpi=100)
            plot_keypoints([p2D_q[valid_q]], colors='lime')
            if SavePlt:
                imq2_path = os.path.join(save_path, 'imq_points')
                if not os.path.exists(imq2_path):
                    os.makedirs(imq2_path, exist_ok=True)
                save_plot(imq2_path + f'/imq_points{idx:06d}.png')
            plt.show()

            plot_images([np.zeros_like(imq)])
            plot_keypoints([p2D_q[valid_q]], colors='lime')
            if SavePlt:
                imq3_path = os.path.join(save_path, 'points')
                if not os.path.exists(imq3_path):
                    os.makedirs(imq3_path, exist_ok=True)
                save_plot(imq3_path + f'/points{idx:06d}.png')
            plt.show()


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
        'max_num_points3D': 5000, # 5000,  # 5000, #both:3976,3D:5000
        'force_num_points3D': False,
        'train_batch_size': 1,
        'test_batch_size': 1,
        'num_workers': 0,
        'satmap_zoom': 18,
        "sampling": 'random',
        "pairs": 10,
        "trans_range": 20,
        "rot_range": 10,
    }

    conf = {
        'normalize_dt': False,
        'optimizer': {'num_iters': 5, },
    }

    dataset_name = 'kitti' # 'kaist_kitti_dense' # 'kitti_voc' #'kaist_kitti_dense' # 'kitti', 'kaist_kitti'
    corruption = 'snow'

    # for kitti, kitti_voc
    # ckpt = '/ws/external/outputs/training/KITTI/resconcat_p5000_reproj2_jac_iters5_t20_r10/checkpoint_best.tar'
    ckpt = '/ws/external/outputs/training/KITTI/LM_p5000_iters5_t20_r10/checkpoint_best.tar'

    # # for kaist_kitti_dense
    # ckpt = '/ws/external/outputs/training/Kaist_Kitti_Dense/resconcat_p5000_jac_iters5_t20_r10_p10_finetune/checkpoint_best.tar'
    # ckpt = '/ws/external/outputs/training/Kaist_Kitti_Dense/LM_p5000_iters5_t20_r10/checkpoint_best.tar'

    experiments_name = ckpt.split('/')[-2]
    # ckpt = '/ws/external/outputs/training/Kaist_Kitti_Dense/resconcat_jac_iters5_t20_r10_p20/checkpoint_best.tar'
    #'/ws/external/outputs/training/Kaist_Kitti_Dense/baseline_geometry.zsn2.l2_v1.0_resconcat_reproj2_jac_iters5/checkpoint_best.tar'
    # ckpt = '/ws/external/outputs/training/NN3d/baseline_geometry.zsn2.l2_v1.0_resconcat_reproj2_jac_iters5/checkpoint_best.tar'
    # '/ws/external/outputs/training/LM_LiDAR_itesr5_40x40_30/checkpoint_best.tar'
    # '/ws/external/outputs/training/NN3d/baseline_geometry.zsn2.l2_v1.0_resconcat_reproj2_jac_iters5/checkpoint_best.tar'
    # '/ws/external/checkpoints/Models/3d_res_embed_aap2_iters5_range.False_dup.False/checkpoint_best.tar'

    if dataset_name == 'kitti_voc':
        save_path = f'/ws/external/visualizations/demo/{dataset_name}/{corruption}/{experiments_name}' # '/ws/external/checkpoints/Models/3d_res_embed_aap2_iters5_range.False_dup.False/visualizations'
    else:
        save_path = f'/ws/external/visualizations/demo/{dataset_name}/{experiments_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.set_grad_enabled(False);
    mpl.rcParams['image.interpolation'] = 'bilinear'

    if dataset_name == 'ford':
        from pixloc.pixlib.datasets.ford import FordAV
        dataset = FordAV(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kitti':
        from pixloc.pixlib.datasets.kitti import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kitti_voc':
        from pixloc.pixlib.datasets.kitti_voc import Kitti_Voc
        dataset = Kitti_Voc(data_conf)
        test_loader = dataset.get_corruption_data_loader(corruption, shuffle=False)
    elif dataset_name == 'kaist_kitti':
        from pixloc.pixlib.datasets.kaist_kitti import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kaist_kitti_dense':
        from pixloc.pixlib.datasets.kaist_kitti_dense import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)

    # val_loader = dataset.get_data_loader('val', shuffle=False)  # or 'train' ‘val’

    # Name of the example experiment. Replace with your own training experiment.
    device = 'cuda'
    # refiner = load_experiment(exp, conf, get_last=True).to(device)
    refiner = load_experiment(exp, conf, ckpt=ckpt).to(device)


    print(OmegaConf.to_yaml(refiner.conf))

    logger = Logger(refiner.optimizer)
    # trainning
    set_seed(20)

    Val(refiner, test_loader, save_path, 0)

    ###################
    ## Load raw data ##
    ###################

    data_conf = {
        'max_num_points3D': 777777,  # 5000,  # 5000, #both:3976,3D:5000
        'force_num_points3D': False,
        'train_batch_size': 1,
        'test_batch_size': 1,
        'num_workers': 0,
        'satmap_zoom': 18,
        "sampling": 'random',
        "pairs": 10,
        "trans_range": 20,
        "rot_range": 10,
    }

    conf = {
        'normalize_dt': False,
        'optimizer': {'num_iters': 5, },
    }


    if dataset_name == 'ford':
        from pixloc.pixlib.datasets.ford import FordAV
        dataset = FordAV(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kitti':
        from pixloc.pixlib.datasets.kitti import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kitti_voc':
        from pixloc.pixlib.datasets.kitti_voc import Kitti_Voc
        dataset = Kitti_Voc(data_conf)
        test_loader = dataset.get_corruption_data_loader(corruption, shuffle=False)
    elif dataset_name == 'kaist_kitti':
        from pixloc.pixlib.datasets.kaist_kitti import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)
    elif dataset_name == 'kaist_kitti_dense':
        from pixloc.pixlib.datasets.kaist_kitti_dense import Kitti
        dataset = Kitti(data_conf)
        test_loader = dataset.get_data_loader('test', shuffle=False)


    # Load_raw(refiner, test_loader, save_path, 0)

