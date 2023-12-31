
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

from pixloc.pixlib.utils.tensor import batch_to_device, map_tensor
from pixloc.pixlib.utils.tools import set_seed
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, plot_matches, cm_RdGn,
    features_to_RGB, add_text, save_plot, plot_valid_points, get_warp_sat2real,
    plot_valid_points2, imsave)

data_conf = {
    'max_num_points3D': 77777,  # 5000, #both:3976,3D:5000
    'force_num_points3D': False,
    'train_batch_size': 1,
    'test_batch_size': 4,
    'num_workers': 0,
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
test_loader = dataset.get_data_loader('test', shuffle=False) #shuffle=True)

# Name of the example experiment. Replace with your own training experiment.
device = 'cuda'
conf = {
    'normalize_dt': False,
    'optimizer': {'num_iters': 5,},
}
# refiner = load_experiment(exp, conf, get_last=True).to(device)
refiner = load_experiment(exp, conf, ckpt='/ws/external/outputs/training/LM_LiDAR1011_e10_i1_b2/checkpoint_best.tar').to(device)
print(OmegaConf.to_yaml(refiner.conf))

class Logger:
    def __init__(self, optimizers=None):
        self.costs = []
        self.dt = []
        self.t = []
        self.camera_trajectory = []
        self.yaw_trajectory = []
        self.pre_q2r = None

        if optimizers is not None:
            for opt in optimizers:
                opt.logging_fn = self.log

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
        self.costs[-1].append(args['cost'].mean(-1).cpu().numpy())
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


logger = Logger(refiner.optimizer)
# trainning
set_seed(20)

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
    for idx, data in zip(range(2959), val_loader):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        cam_r = data['ref']['camera']
        cam_q = data['query']['camera']
        p3D_q = data['query']['points3D']

        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam']*p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        p2D_r_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        p2D_r_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)

        # 2d
        p2D_q, visible = cam_q.world2image(data['query']['T_w2cam'] * p3D_q)
        from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear
        F_q, _ = interpolate_tensor_bilinear(data['query']['image'], p2D_q)

        p3D_ref, visible = cam_r.world2image3d_(data['T_q2r_gt'] * p3D_q)
        F_q2r = cam_r.voxelize_(p3D_ref.unsqueeze(0), F_q.unsqueeze(0), size=(1, 3, 20, 1280, 1280), level=0)
        from pixloc.visualization.viz_2d import imsave
        imsave(F_q2r[0], save_path, 'validgt_g2s_points2d')
        # import matplotlib
        # plot_images([F_q2r[0].permute(1, 2, 0)], cmaps=matplotlib.cm.gnuplot2, titles=['2d'])
        # if SavePlt:
        #     save_plot(save_path + f'/lidargt_g2s_points2d.png')

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
            plot_images([imr],dpi=50,  # set to 100-200 for higher res
                             titles=[(valid_r.sum().item(), valid_q.sum().item()), errP + errt])
            plot_keypoints([p2D_r_gt[valid]], colors='lime')
            plot_keypoints([p2D_r_init[valid]], colors='red')
            plot_keypoints([p2D_r_opt[valid]], colors='blue')
            if SavePlt:
                save_plot(save_path + f'/sat_points.png')
            plt.show()
            plot_images([imq], dpi=100)
            plot_keypoints([p2D_q[valid_q]], colors='lime')
            if SavePlt:
                save_plot(save_path + f'/ground_points.png')
            # add_text(0, 'reference')
            # add_text(1, 'query')
            plt.show()

            imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)

            # g2s with GH and T
            imr1 = imr.permute(2, 0, 1)
            imq1 = imq.permute(2, 0, 1)
            c, a, a = imr1.size()
            c, h, w = imq1.size()
            uv1 = get_warp_sat2real(imr1)
            uv1 = data['T_q2r_gt'].cuda().inv() * uv1
            # uv, mask = cam_query.world2image(uv1)
            uv, mask = data['query']['camera'].cuda().world2image(uv1)

            scale = torch.tensor([w - 1, h - 1]).to(uv)
            uv = (uv / scale) * 2 - 1
            uv = uv.clamp(min=-2, max=2)  # ideally use the mask instead
            g2s = torch.nn.functional.grid_sample(
                imq1.cuda().unsqueeze(dim=0), uv.unsqueeze(dim=0), mode='bilinear', align_corners=True)

            validgt_sat_points = plot_valid_points2(imr, imr, p2D_r_gt[valid], p2D_r_gt[valid])
            imsave(validgt_sat_points.permute((2, 0, 1)), save_path, 'validgt_sat_points')
            validgt_g2s_points = plot_valid_points2(imr, imq, p2D_r_gt[valid], p2D_q[valid])
            imsave(validgt_g2s_points.permute((2, 0, 1)), save_path, 'validgt_g2s_points')
            validgt_lidar_points = plot_valid_points2(imr, torch.ones_like(imr), p2D_r_gt[valid], p2D_r_gt[valid])
            imsave(validgt_lidar_points.permute((2, 0, 1)), save_path, 'validgt_lidar_points')

            validinit_g2s_points = plot_valid_points2(imr, imq, p2D_r_init[valid], p2D_q[valid])
            imsave(validinit_g2s_points.permute((2, 0, 1)), save_path, 'validinit_g2s_points')

            imsave(imr.permute((2, 0, 1)), save_path, 'sat')
            imsave(imq.permute((2, 0, 1)), save_path, 'grd')

            # plot_images([g2s[0].permute(1, 2, 0).cpu()], dpi=100)
            # if SavePlt:
            #     save_plot(save_path + f'/gt_g2s.png')
            #
            # plot_valid_points(imr, imr, p2D_r_gt[valid], p2D_r_gt[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validgt_sat_points.png')

            # plot_valid_points(imr, imr, p2D_r_opt[valid], p2D_r_opt[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validopt_sat_points.png')
            #
            # plot_valid_points(imr, imr, p2D_r_init[valid], p2D_r_init[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validinit_sat_points.png')
            #
            # plot_valid_points(imr, imq, p2D_r_gt[valid], p2D_q[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validgt_g2s_points.png')
            #
            # plot_valid_points(imr, imq, p2D_r_init[valid], p2D_q[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validinit_g2s_points.png')
            #
            # plot_valid_points(imr, imq, p2D_r_opt[valid], p2D_q[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/validopt_g2s_points.png')
            #
            # plot_valid_points(imr, torch.ones_like(imr), p2D_r_gt[valid], p2D_r_gt[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/lidargt_sat_points.png')
            #
            # plot_valid_points(imr, torch.ones_like(imr), p2D_r_gt[valid], p2D_q[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/lidargt_g2s_points.png')
            #
            # plot_valid_points(imr, torch.ones_like(imr), p2D_r_init[valid], p2D_r_init[valid])
            # if SavePlt:
            #     save_plot(save_path + f'/lidarinit_sat_points.png')

            # feature & confidence

            # for i, (F0, F1) in enumerate(zip(pred['ref']['feature_maps'], pred['query']['feature_maps'])):
            #     C_r, C_q = pred['ref']['confidences'][i][0], pred['query']['confidences'][i][0]
            #     C_r = min_max_norm(C_r)
            #     C_q = min_max_norm(C_q)
            #     plot_images([C_r], cmaps=mpl.cm.gnuplot2, dpi=50)
            #     axes = plt.gcf().axes
            #     axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)
            #     if SavePlt:
            #         save_plot(save_path + f'/c_s_0_{i}.png')
            #     plt.show()
            #     plot_images([C_q], cmaps=mpl.cm.gnuplot2, dpi=50)
            #     axes = plt.gcf().axes
            #     axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)
            #     if SavePlt:
            #         save_plot(save_path + f'/c_g_0_{i}.png')
            #     plt.show()
            #
            #     print(F0.dtype, torch.min(F0), torch.max(F0))
            #     print(F1.dtype, torch.min(F1), torch.max(F1))
            #     plot_images(features_to_RGB(F0.numpy(), skip=1), dpi=50)
            #     if SavePlt:
            #         save_plot(save_path + f'/f_s_{i}.png')
            #     plt.show()
            #     plot_images(features_to_RGB(F1.numpy(), skip=1), dpi=50)
            #     if SavePlt:
            #         save_plot(save_path + f'/f_g_{i}.png')
            #     plt.show()
            #     # plot_images(F0.numpy(), dpi=50)
            #     # if SavePlt:
            #     #     save_plot(save_path + f'/f_s_{i}.png')
            #     # plt.show()
            #     # plot_images(F1.numpy(), dpi=50)
            #     # if SavePlt:
            #     #     save_plot(save_path + f'/f_g_{i}.png')
            #     # plt.show()
            #
            # costs = logger.costs
            # fig, axes = plt.subplots(1, len(costs), figsize=(len(costs)*4.5, 4.5))
            # for i, (ax, cost) in enumerate(zip(axes, costs)):
            #     ax.plot(cost) if len(cost)>1 else ax.scatter(0., cost)
            #     ax.set_title(f'({i}) Scale {i//3} Level {i%3}')
            #     ax.grid()
            # plt.show()
            #
            # colors = mpl.cm.cool(1 - np.linspace(0, 1, len(logger.camera_trajectory)))[:, :3]
            # plot_images([imr])
            # axes = plt.gcf().axes
            # for i, (p2s, _), (p2e, _), T, c in zip(range(len(logger.camera_trajectory)), logger.camera_trajectory, logger.yaw_trajectory, logger.t, colors):
            #     # plot the direction of the body frame
            #     if i == 0:
            #         start_0 = p2s
            #         end_0 = p2e
            #         axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color='r')
            #     elif i == len(logger.camera_trajectory)-1:
            #         axes[0].quiver(p2s[:, 0], p2s[:, 1], p2e[:, 0] - p2s[:, 0], p2s[:, 1] - p2e[:, 1], color='b')
            #         axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color='r')
            #     else:
            #         axes[0].quiver(p2s[:,0], p2s[:,1], p2e[:,0]-p2s[:,0], p2s[:,1]-p2e[:,1], color=c[None])
            # axes[0].quiver(logger.camera_gt[:, 0], logger.camera_gt[:, 1], logger.camera_gt_yaw[:, 0]-logger.camera_gt[:, 0],
            #                logger.camera_gt[:, 1]-logger.camera_gt_yaw[:, 1], color='lime')
            # logger.clear_trajectory()
            # if SavePlt:
            #     save_plot(save_path+'/pose_refine.png')
            # plt.show()

    acc = acc/cnt
    print('acc of a epoch:#####',acc)
    if acc > best_result:
        print('best acc:@@@@@', acc)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(refiner.state_dict(), save_path + 'Model_best.pth')
    return acc

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

    save_path = '2d_1211'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 0: # test
        test(refiner, test_loader) #val_loader
    if 1: # visualization
        Val(refiner, val_loader, save_path, 0)

