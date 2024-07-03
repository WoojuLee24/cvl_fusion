# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:44:19 2020

@author: loocy
"""

from pixloc.pixlib.datasets.base_dataset import BaseDataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pixloc.pixlib.datasets.Kitti_utils as Kitti_utils
from matplotlib import pyplot as plt
# from sklearn.neighbors import NearestNeighbors
import kitti_data_process.Kitti_gps_coord_func as gps_func
import random
import cv2
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.augmix_dataset import AugMixDataset
from pixloc.pixlib.models.pointnet2 import farthest_point_sample
# from pytorch3d.ops import sample_farthest_points

root_dir = "/ws/data/gazebo_kitti" # your kitti dir
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'
oxts_dir = 'oxts/data'
vel_dir = 'velodyne_points/data'

grd_ori_size = (375, 1242) # different size in Kitti 375×1242, 370×1224,374×1238, and376×1241, but 375*1242 in calibration
grd_process_size = (375, 1242) # (1200, 1920) # (384, 1248)
satellite_ori_size = 1280
meter_per_pixel = 0.078

ToTensor = transforms.Compose([
    transforms.ToTensor()])

class Kitti(BaseDataset):
    default_conf = {
        'two_view': True,
        'max_num_points3D': 5000,
        'center_num_points3D': 1024,
        'force_num_points3D': False,
        'rot_range': 15,
        'trans_range': 5,
        'sampling': 'random',
        'pose_from': 'aa',
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        #assert split != 'test', 'Not supported'

        dataset = _Dataset(self.conf, split)

        # augmentation
        if split == 'train' and 'augmix' in self.conf.aug:
            dataset = AugMixDataset(args=self.conf, dataset=dataset, severity=self.conf.aug_severity)

        return dataset

def read_calib(calib_file_name, camera_id='rect_02'):
    with open(calib_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # left color camera k matrix
            if 'P_'+camera_id in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')

                camera_P = np.asarray([float(i) for i in valus]).reshape((3, 4))
                fx = float(valus[0])
                cx = float(valus[2])
                fy = float(valus[5])
                cy = float(valus[6])

                camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                camera_k = np.asarray(camera_k)
                tx, ty, tz = float(valus[3]), float(valus[7]), float(valus[11])
                camera_t = np.linalg.inv(camera_k) @ np.asarray([tx, ty, tz])
                camera_ex = np.hstack((np.eye(3), camera_t.reshape(3,1)))
                camera_ex = np.vstack((camera_ex, np.array([0,0,0,1])))
            #  add R_rect_00: P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X
            if 'R_rect_00' in line:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                camera_R0 = np.asarray([float(i) for i in valus]).reshape((3, 3))
                camera_R0 = np.hstack((camera_R0, np.zeros([3, 1])))
                camera_R0 = np.vstack((camera_R0, np.array([0, 0, 0, 1])))
        camera_ex = camera_ex @ camera_R0

    return camera_k, camera_ex


def read_sensor_rel_pose(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # left color camera k matrix
            if 'R' in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                R = np.asarray([float(i) for i in valus]).reshape((3, 3))

            if 'T' in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                T = np.asarray([float(i) for i in valus]).reshape((3))
    rel_pose_hom = np.eye(4)
    rel_pose_hom[:3, :3] = R
    rel_pose_hom[:3, 3] = T

    return R, T, rel_pose_hom



def project_lidar_to_cam(velodyne, camera_k, lidar2cam_R, lidar2cam_T, img_size):

    points3d = lidar2cam_R @ velodyne.T[:3, :] + lidar2cam_T[:, None]
    velo_mask = points3d[2, :] > 0
    points3d = points3d[:, velo_mask]
    velodyne_trimed = velodyne.T[:, velo_mask]

    # project the points to the camera
    points2d = camera_k @ points3d
    points2d = points2d[:2, :] / points2d[2, :]

    u_mask = (points2d[0] < img_size[1]) & (points2d[0] > 0)
    v_mask = (points2d[1] < img_size[0]) & (points2d[1] > 0)
    uv_mask = u_mask & v_mask
    return points2d[:, uv_mask], points3d[:, uv_mask], velodyne_trimed[:, uv_mask]

class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.root = root_dir
        self.conf = conf
        self.satmap_dir = 'satmap'
        self.sat_pair = np.load(os.path.join(self.root, grdimage_dir, 'groundview_satellite_pair.npy'), allow_pickle=True)

        # read form txt files
        self.file_name = []
        txt_file_name = os.path.join(self.root, grdimage_dir, 'split', split+'_files.txt')
        with open(txt_file_name, "r") as txt_f:
            lines = txt_f.readlines()
            for line in lines:
                line = line.strip()
                # check grb file exist
                grb_file_name = os.path.join(self.root, grdimage_dir, line[:-14], left_color_camera_dir, line[-14:])
                if not os.path.exists(grb_file_name):
                    # ignore frames with out velodyne
                    print(grb_file_name + ' do not exist!!!')
                    continue

                velodyne_file_name = os.path.join(self.root, grdimage_dir, line[:-14], vel_dir,
                                                  line[-14:].lower().replace('.png', '.bin'))
                if not os.path.exists(velodyne_file_name):
                    # ignore frames with out velodyne
                    print(velodyne_file_name + ' do not exist!!!')
                    continue

                self.file_name.append(line)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name
        file_name = self.file_name[idx]
        image_no = file_name

        # get calibration information, do not adjust image size change here
        camera_k, camera_ex = read_calib(
            os.path.join(self.root, grdimage_dir, 'calib_cam_to_cam.txt'), 'rect_00')
        pose_camera_ex = Pose.from_4x4mat(camera_ex)
        imu2lidar_R, imu2lidar_T, imu2lidar_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, 'calib_imu_to_velo.txt'))
        pose_imu2lidar = Pose.from_4x4mat(imu2lidar_H)
        lidar2cam_R, lidar2cam_T, lidar2cam_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, 'calib_velo_to_cam.txt'))
        pose_lidar2cam = pose_camera_ex @ Pose.from_4x4mat(lidar2cam_H)
        # computer the relative pose between imu to camera
        imu2camera = pose_lidar2cam.compose(pose_imu2lidar)# pose_imu2lidar.inv().compose(pose_lidar2cam.inv())
        camera_center_loc = -imu2camera.transform(np.array([0.,0.,0.]))

        # get location & rotation
        oxts_file_name = os.path.join(self.root, grdimage_dir, image_no[:-14], oxts_dir,
                                      image_no[-14:].lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
        location = [float(content[0]), float(content[1]), float(content[2])]
        roll, pitch, heading = float(content[3]), float(content[4]), float(content[5])

        # read lidar points
        velodyne_file_name = os.path.join(self.root, grdimage_dir, image_no[:-14], vel_dir,
                                          image_no[-14:].lower().replace('.png', '.bin'))
        velodyne = np.fromfile(velodyne_file_name, dtype=np.float32).reshape(-1, 4)
        velodyne[:, 3] = 1.0

        # ground images, left color camera
        left_img_name = os.path.join(self.root, grdimage_dir, image_no[:-14], left_color_camera_dir, image_no[-14:].lower())
        with Image.open(left_img_name, 'r') as GrdImg:
            grd_left = GrdImg.convert('RGB')
            grd_ori_H = grd_left.size[1]
            grd_ori_W = grd_left.size[0]
            # resize
            grd_left = transforms.functional.resize(grd_left, grd_process_size)
            # process camera_k for resize
            camera_k[0] *=  grd_process_size[1]/grd_ori_size[1]
            camera_k[1] *=  grd_process_size[0]/grd_ori_size[0]

            grd_left = ToTensor(grd_left)

        # project these lidar points to camera coordinate
        # velodyne_local: lidar points (3D) which are visible to the image, in the velodyne coordinate system
        _, cam_3d, _ = project_lidar_to_cam(velodyne, camera_k, lidar2cam_R, lidar2cam_T, (grd_ori_H,grd_ori_W))

        # grd left
        camera_para = (camera_k[0,0],camera_k[1,1],camera_k[0,2],camera_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(grd_process_size[1]), height=int(grd_process_size[0])))
        grd_image = {
            'image': grd_left.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float(),  # already consider calibration in points3D
        }

        if self.conf.pose_from == 'aa':
            grd2imu = Pose.from_aa(np.array([-roll, pitch, -heading]), np.zeros(3)) # grd_x:east, grd_y:north, grd_z:up
            grd2cam = imu2camera@grd2imu
            grd2sat = np.array([[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # grd z->-sat z; grd x->sat x, grd y->-sat y
            grd2sat = Pose.from_4x4mat(grd2sat)
            q2r_gt = grd2sat @ (grd2cam.inv())
        elif self.conf.pose_from == 'rt':
            R = np.array([[np.cos(-heading), -np.sin(-heading), 0], [np.sin(-heading), np.cos(-heading), 0], [0, 0, 1.]])
            t = np.zeros(3)
            grd2imu = Pose.from_Rt(R, t)
            grd2cam = imu2camera @ grd2imu
            grd2sat = np.array([[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # grd z->-sat z; grd x->sat x, grd y->-sat y
            grd2sat = Pose.from_4x4mat(grd2sat)
            q2r_gt = grd2sat @ (grd2cam.inv())

        # satellite map
        SatMap_name = self.sat_pair.item().get(os.path.join(file_name))
        extension = SatMap_name.split('.')[-1]
        SatMap_name = '.'.join(SatMap_name.split('.')[:-1])

        sat_gps = SatMap_name.split('_')
        sat_gps = [float(sat_gps[3]), float(sat_gps[5])]
        SatMap_name = os.path.join(root_dir, self.satmap_dir, '.'.join([SatMap_name, extension]))
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # get ground-view, satellite image shift
        x_sg, y_sg = gps_func.angular_distance_to_xy_distance_v2(sat_gps[0], sat_gps[1], location[0],
                                                                 location[1])
        x_sg = int(x_sg / meter_per_pixel)
        y_sg = int(-y_sg / meter_per_pixel)

        # add the offset between camera and body to shift the center to query camera
        cam_pixel = q2r_gt*camera_center_loc
        cam_location_x = x_sg + satellite_ori_size / 2.0 + cam_pixel[0, 0]
        cam_location_y = y_sg + satellite_ori_size / 2.0 + cam_pixel[0, 1]

        # sat
        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE',
            params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0, 0, 0, 0, np.infty),
            # np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))
        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float()  # grd 2 sat in q2r, so just eye(4)
        }
        # project to sat and find visible mask
        cam_3d = cam_3d.T
        _, visible = camera.world2image(torch.from_numpy(cam_3d).float())
        cam_3d = cam_3d[visible]
        key_points = torch.from_numpy(cam_3d).float()
        grd_image['points3D_type'] = 'lidar'

        # max distance is 240 meter
        mask = key_points[:, 2] < 100.0
        key_points = key_points[mask]

        num_diff = self.conf.max_num_points3D - len(key_points)
        if num_diff < 0:
            # select max_num_points
            if self.conf.sampling == 'random':
                sample_idx = np.random.choice(range(len(key_points)), self.conf.max_num_points3D)
                key_points = key_points[sample_idx]
            elif self.conf.sampling == 'distance':
                distance = key_points.pow(2).mean(dim=-1)
                sorted, indices = torch.sort(distance, dim=0, descending=True)
                sample_idx = indices[:self.conf.max_num_points3D]
                key_points = key_points[sample_idx]
            # elif self.conf.sampling == 'random_fps':
            #     rand_idx = torch.randperm(key_points.size(0))
            #     key_points = key_points[rand_idx].unsqueeze(dim=0)
            #     points_per_batch = torch.tensor([self.conf.max_num_points3D]).to(key_points.device)
            #     key_points, sample_idx = sample_farthest_points(key_points, points_per_batch, self.conf.max_num_points3D)
            #     key_points = key_points.squeeze(dim=0)
            elif self.conf.sampling == 'random_fps':
                sample_idx = np.random.choice(range(len(key_points)), self.conf.max_num_points3D)
                key_points = key_points[sample_idx]
                key_points = key_points.unsqueeze(dim=0)
                # key_points, sample_idx = sample_farthest_points(key_points, points_per_batch, self.conf.max_num_points3D)
                centroid_idx = farthest_point_sample(key_points, self.conf.center_num_points3D)
                key_points = key_points.squeeze()
                centroid_idx = centroid_idx.squeeze()
                grd_image['centroid_idx'] = centroid_idx
                # key_points = key_points[centroid_idx].squeeze(dim=0)
                # key_points = key_points.squeeze(dim=0)
            elif self.conf.sampling == 'fps':
                key_points = key_points.unsqueeze(dim=0)
                centroid_idx = farthest_point_sample(key_points, self.conf.max_num_points3D) # self.conf.center_num_points3D
                key_points = key_points.squeeze()
                centroid_idx = centroid_idx.squeeze()
                key_points = key_points[centroid_idx]
                grd_image['centroid_idx'] = centroid_idx
                # key_points = key_points[centroid_idx].squeeze(dim=0)
                # key_points = key_points.squeeze(dim=0)
        elif num_diff > 0 and self.conf.force_num_points3D:
            point_add = torch.ones((num_diff, 3)) * key_points[-1]
            key_points = torch.cat([key_points, point_add], dim=0)
        grd_image['points3D'] = key_points

        # ramdom shift translation and ratation on yaw
        YawShiftRange = self.conf.rot_range * np.pi / 180  # 15 * np.pi / 180  # SIBCL: 15 degree, cvl: 10 degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        R_yaw = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        TShiftRange = self.conf.trans_range # 5  # SIBCL: 5 meter, cvl: 10 meter
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
        # T[0] = 0 # debug: no shift on lon
        # T[1] = 0  # debug: no shift on lat
        T[2] = 0  # no shift on height

        shift = Pose.from_Rt(R_yaw, T)
        q2r_init = shift @ q2r_gt
        shift_gt = np.array([T[0], T[1], yaw])
        shift_range = np.array([TShiftRange, TShiftRange, YawShiftRange], dtype=np.float32)

        # statistics
        mean = np.array([[0, 0, 95.3919]], dtype=np.float32)
        std = np.array([[22.4492, 6.3773, 36.7385]], dtype=np.float32)
        # mean = np.array([[-6.9251, -4.0433, 95.3919]])
        # std = np.array([[22.4492, 6.3773, 36.7385]])

        data = {
            'ref': sat_image,
            'query': grd_image,
            'T_q2r_init': q2r_init.float(),
            'T_q2r_gt': q2r_gt.float(),
            'shift_gt': shift_gt,
            'shift_range': shift_range,
            'mean': mean,
            'std': std
        }

        # debug
        if 0:
            image = transforms.functional.to_pil_image(grd_left, mode='RGB')
            image.save(f'/ws/data/kaist_mobile/debug_images/grd_{idx}.png')
            image = transforms.functional.to_pil_image(sat_map, mode='RGB')
            image.save(f'/ws/data/kaist_mobile/debug_images/sat_{idx}.png')
        if 0:
            def distance_to_color(distance, min_distance, max_distance):
                # Normalize the distance to be within [0, 1]
                norm_distance = (distance - min_distance) / (max_distance - min_distance)
                # Convert the normalized distance to a color (closer: red, farther: green)
                r = int(255 * (norm_distance))
                g = int(255 * (1 - norm_distance))
                b = 0
                return (r, g, b)

            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            color_image0 = transforms.functional.to_pil_image(grd_left, mode='RGB')
            color_image0 = np.array(color_image0)
            # debug satellite map
            color_image1 = transforms.functional.to_pil_image(sat_map, mode='RGB')
            color_image1 = np.array(color_image1)

            grd_2d, _ = grd_image['camera'].world2image(grd_image['points3D']) ##camera 3d to 2d
            grd_2d = grd_2d.T
            # for j in range(grd_2d.shape[1]):
            #     cv2.circle(color_image0, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 5, (0, 255, 0),
            #            -1)

            distances = grd_image['points3D'][:, 2]
            min_distance = torch.min(distances)
            max_distance = torch.max(distances)

            for j in range(grd_2d.shape[1]):
                distance = distances[j]
                color = distance_to_color(distance, min_distance, max_distance)
                cv2.circle(color_image0, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 5, color, -1)

            # sat gt green
            sat_3d = data['T_q2r_gt']*grd_image['points3D']
            sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            sat_2d = sat_2d.T
            for j in range(sat_2d.shape[1]):
                cv2.circle(color_image1, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (0, 255, 0),
                       -1)

            origin = torch.zeros(3)
            origin_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * origin)
            cv2.circle(color_image1, (np.int32(origin_2d_gt[0,0]), np.int32(origin_2d_gt[0,1])), 5, (0, 255, 0),
                       -1)

            #sat init red
            sat_3d = data['T_q2r_init']* grd_image['points3D']
            sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            sat_2d = sat_2d.T
            for j in range(sat_2d.shape[1]):
                cv2.circle(color_image1, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (255, 0, 0),
                       -1)

            ax1.imshow(color_image0)
            ax2.imshow(color_image1)

            # camera position
            # camera gt position
            origin = torch.zeros(3)
            origin_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * origin)
            origin_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * origin)
            direct = torch.tensor([0,0,6.])
            direct_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * direct)
            direct_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * direct)
            origin_2d_gt = origin_2d_gt.squeeze(0)
            origin_2d_init = origin_2d_init.squeeze(0)
            direct_2d_gt = direct_2d_gt.squeeze(0)
            direct_2d_init = direct_2d_init.squeeze(0)

            # plot the init direction of the body frame
            plt.scatter(x=origin_2d_init[0], y=origin_2d_init[1], c='r', s=10)
            plt.quiver(origin_2d_init[0], origin_2d_init[1], direct_2d_init[0] - origin_2d_init[0],
                       origin_2d_init[1] - direct_2d_init[1], color=['r'], scale=None)
            # plot the gt direction of the body frame
            plt.scatter(x=origin_2d_gt[0], y=origin_2d_gt[1], c='g', s=10)
            plt.quiver(origin_2d_gt[0], origin_2d_gt[1], direct_2d_gt[0] - origin_2d_gt[0],
                       origin_2d_gt[1] - direct_2d_gt[1], color=['g'], scale=None)
            # plt.show()
            plt.savefig(f'/ws/data/kaist_mobile/debug_images/pointcloud_{idx}.png')
            print(idx,file_name, pitch, roll)

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'max_num_points3D': 1024,
        'force_num_points3D': True,
        'batch_size': 1,
        'seed': 1,
        'num_workers': 0,
    }
    dataset = Kitti(conf)
    loader = dataset.get_data_loader('train', shuffle=False)  # or 'train' ‘val’

    for it, data in enumerate(loader):
        print(it)

