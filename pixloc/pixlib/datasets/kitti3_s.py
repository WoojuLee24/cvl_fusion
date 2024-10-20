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
import torchvision.transforms.functional as TF
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.augmix_dataset import AugMixDataset
from pixloc.pixlib.models.pointnet2 import farthest_point_sample
# from pytorch3d.ops import sample_farthest_points
from pixloc.pixlib.geometry.wrappers import project_grd_to_map, project_map_to_grd


root_dir = "/ws/data/kitti-vo" # your kitti dir
satmap_zoom = 18
satmap_dir = 'satmap_'+str(satmap_zoom)
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'
oxts_dir = 'oxts/data'
vel_dir = 'velodyne_points/data'

grd_ori_size = (375, 1242) # different size in Kitti 375×1242, 370×1224,374×1238, and376×1241, but 375*1242 in calibration
grd_process_size = (384, 1248)
satellite_ori_size = 1280
satellite_process_size = 512

CameraGPS_shift_left = [1.08, 0.26]
CameraGPS_shift_right = [1.08, 0.8]  # 0.26 + 0.54

ToTensor = transforms.Compose([
    transforms.ToTensor()])

class Kitti(BaseDataset):
    default_conf = {
        'two_view': True,
        'max_num_points3D': 5000,
        'max_num_out_points3D': 15000,
        'center_num_points3D': 1024,
        'force_num_points3D': True,
        'rot_range': 15,
        'trans_range': 5,
        'satmap_zoom': 18,
        'sampling': 'random',
        'pose_from': 'rt',
        'part': 0.5,
        # 'project': 2,
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
    # points3d = points3d[:, velo_mask]
    # velodyne_trimed = velodyne.T[:, velo_mask]

    # project the points to the camera
    points2d = camera_k @ points3d
    points2d = points2d[:2, :] / points2d[2, :]

    u_mask = (points2d[0] < img_size[1]) & (points2d[0] > 0)
    v_mask = (points2d[1] < img_size[0]) & (points2d[1] > 0)
    uv_mask = u_mask & v_mask

    mask = uv_mask & velo_mask

    return points2d[:, mask], points3d[:, uv_mask], points3d, mask # velodyne_trimed[:, uv_mask]

class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.root = root_dir
        self.conf = conf
        self.satmap_zoom = self.conf.satmap_zoom
        self.satmap_dir = 'satmap_'+str(self.satmap_zoom)+'_all'
        self.sat_pair = np.load(os.path.join(self.root, grdimage_dir, 'groundview_satellite_pair_'+str(self.satmap_zoom)+'_all.npy'), allow_pickle=True)

        # read form txt files
        self.file_name = []
        txt_file_name = os.path.join(self.root, grdimage_dir, 'kitti_split', split+'_files.txt')
        with open(txt_file_name, "r") as txt_f:
            lines = txt_f.readlines()
            for line in lines:
                line = line.strip()
                # check grb file exist
                grb_file_name = os.path.join(self.root, grdimage_dir, line[:38], left_color_camera_dir,
                                                  line[38:].lower())
                if not os.path.exists(grb_file_name):
                    # ignore frames with out velodyne
                    print(grb_file_name + ' do not exist!!!')
                    continue

                velodyne_file_name = os.path.join(self.root, grdimage_dir, line[:38], vel_dir,
                                                  line[38:].lower().replace('.png', '.bin'))
                if not os.path.exists(velodyne_file_name):
                    # ignore frames with out velodyne
                    print(velodyne_file_name + ' do not exist!!!')
                    continue

                self.file_name.append(line)

        if self.conf.part < 1.0:
            # print("part: ", self.conf.part)
            indexes = int(len(self.file_name) * self.conf.part)
            self.file_name = self.file_name[:indexes]

    def __len__(self):
        return len(self.file_name)

    def sample_points(self, key_points, max_points):
        num_diff = max_points - len(key_points)
        if num_diff < 0:
            sample_idx = np.random.choice(range(len(key_points)), max_points)
            key_points = key_points[sample_idx]
        elif num_diff > 0:
            point_add = torch.ones((num_diff, 3)) * key_points[-1]
            key_points = torch.cat([key_points, point_add], dim=0)

        return key_points

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name
        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # get calibration information, do not adjust image size change here
        camera_k, camera_ex = read_calib(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt'))
        pose_camera_ex = Pose.from_4x4mat(camera_ex)
        imu2lidar_R, imu2lidar_T, imu2lidar_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_imu_to_velo.txt'))
        pose_imu2lidar = Pose.from_4x4mat(imu2lidar_H)
        lidar2cam_R, lidar2cam_T, lidar2cam_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_velo_to_cam.txt'))
        pose_lidar2cam = pose_camera_ex @ Pose.from_4x4mat(lidar2cam_H)
        # computer the relative pose between imu to camera
        imu2camera = pose_lidar2cam.compose(pose_imu2lidar)# pose_imu2lidar.inv().compose(pose_lidar2cam.inv())
        camera_center_loc = -imu2camera.transform(np.array([0.,0.,0.]))

        # get location & rotation
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
        location = [float(content[0]), float(content[1]), float(content[2])]
        roll, pitch, heading = float(content[3]), float(content[4]), float(content[5])

        # read lidar points
        velodyne_file_name = os.path.join(self.root, grdimage_dir, drive_dir, vel_dir,
                                      image_no.lower().replace('.png', '.bin'))
        velodyne = np.fromfile(velodyne_file_name, dtype=np.float32).reshape(-1, 4)
        velodyne[:, 3] = 1.0

        # ground images, left color camera
        left_img_name = os.path.join(self.root, grdimage_dir, drive_dir, left_color_camera_dir, image_no.lower())
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
        _, cam_3d_, cam_3d, cam_mask = project_lidar_to_cam(velodyne, camera_k, lidar2cam_R, lidar2cam_T, (grd_ori_H,grd_ori_W))

        # grd left
        camera_para = (camera_k[0,0],camera_k[1,1],camera_k[0,2],camera_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(grd_process_size[1]), height=int(grd_process_size[0])))
        grd_image = {
            'image': grd_left.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float(),  # already consider calibration in points3D
            'camera_h': torch.tensor(1.65),

        }

        # satellite map
        SatMap_name = self.sat_pair.item().get(file_name)
        sat_gps = SatMap_name.split('_')
        sat_gps = [float(sat_gps[3]), float(sat_gps[5])]
        SatMap_name = os.path.join(root_dir, self.satmap_dir, SatMap_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            # sat_map = ToTensor(sat_map)

        # align the satellite-view image
        meter_per_pixel = Kitti_utils.get_meter_per_pixel(self.satmap_zoom, scale=1)
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / meter_per_pixel),
                                          resample=Image.BILINEAR)

        # ramdom shift translation and rotation
        YawShiftRange = self.conf.rot_range  # 15 * np.pi / 180  # SIBCL: 15 degree, cvl: 10 degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        # yaw = 2 * YawShiftRange * 1 - YawShiftRange
        yaw_rad = yaw * np.pi / 180
        R_yaw = torch.tensor([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
        TShiftRange = self.conf.trans_range  # 5  # SIBCL: 5 meter, cvl: 10 meter
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange # [lon, lat, height]
        # T = 2 * TShiftRange * np.ones((3)) - TShiftRange # [lon, lat, height]
        T[2] = 0  # no shift on height

        sat_rand_shift = sat_align_cam.transform(sat_align_cam.size, Image.AFFINE,
                                                 (1, 0, -T[0] / meter_per_pixel, 0, 1, -T[1] / meter_per_pixel), resample=Image.BILINEAR)
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(-yaw)

        # # debug
        # from pixloc.visualization.viz_2d import imsave
        # save_dir = '/ws/external/debug_images/kitti3_s'
        # imsave(sat_map, save_dir, 'sat_1orig')
        # imsave(sat_rot, save_dir, 'sat_2rot')
        # imsave(sat_align_cam, save_dir, 'sat_3align')
        # imsave(sat_rand_shift, save_dir, 'sat_4rsfhit')
        # imsave(sat_rand_shift_rand_rot, save_dir, 'sat_5rshift_rrot')

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, satellite_process_size) # TODO
        # imsave(sat_map, save_dir, 'sat_6final')
        sat_map = ToTensor(sat_map)


        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])
        t = np.zeros(3)
        grd2imu = Pose.from_Rt(R, t)
        grd2cam = imu2camera @ grd2imu
        grd2sat = np.array(
            [[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # grd z->-sat z; grd x->sat x, grd y->-sat y
        grd2sat = Pose.from_4x4mat(grd2sat)
        q2r_init = grd2sat @ (grd2cam.inv())
        # q2r_init.t[-1] = 0

        shift = Pose.from_Rt(R_yaw, T)
        q2r_gt = shift @ q2r_init   # TODO: metrics should be changed
        shift_gt = np.array([T[0], T[1], yaw_rad])
        shift_range = np.array([TShiftRange, TShiftRange, YawShiftRange], dtype=np.float32)

        # apply offset on the ref camera
        cam_location_x = satellite_process_size / 2.0
        cam_location_y = satellite_process_size / 2.0

        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE',
            params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0, 0, 0, 0, np.infty),
            # np.infty for parallel projection
            width=int(satellite_process_size), height=int(satellite_process_size)))
        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float()  # grd 2 sat in q2r, so just eye(4)
        }
        # project to sat and find visible mask
        cam_3d = cam_3d.T
        _, visible = camera.world2image(torch.from_numpy(cam_3d).float())
        cam_mask = torch.from_numpy(cam_mask) & visible

        # cam_3d = cam_3d[visible]
        key_points = torch.from_numpy(cam_3d).float()
        grd_image['points3D_type'] = 'lidar3'

        # new code
        cam_3d_visible = self.sample_points(key_points[cam_mask], self.conf.max_num_points3D)
        cam_3d_invisible = self.sample_points(key_points[~cam_mask], self.conf.max_num_out_points3D)
        key_points = torch.cat([cam_3d_visible, cam_3d_invisible], dim=0)
        mask = torch.zeros(key_points.size(0), dtype=torch.bool)
        mask[:self.conf.max_num_points3D] = True

        grd_image['points3D'] = key_points  # all
        grd_image['points3D_mask'] = mask  # only visible

        # statistics
        mean = np.array([[-0.1917,  0.9250, 15.6600]], dtype=np.float32)
        std = np.array([[6.9589,  0.8642, 11.5166]], dtype=np.float32)

        # scene
        scene = drive_dir[:4] + drive_dir[5:7] + drive_dir[8:10] + drive_dir[28:32] + image_no[:10]
        normal = torch.tensor([[0., 1, 0.]])  # down, y axis of camera coordinate

        data = {
            'ref': sat_image,
            'query': grd_image,
            'T_q2r_init': q2r_init.float(),
            'T_q2r_gt': q2r_gt.float(),
            'shift_gt': shift_gt,
            'shift_range': shift_range,
            'mean': mean,
            'std': std,
            'normal': normal,
        }

        # debug
        if 0:
            image = transforms.functional.to_pil_image(grd_left, mode='RGB')
            image.save('/ws/external/debug_images/kitti2/grd.png')
            image = transforms.functional.to_pil_image(sat_map, mode='RGB')
            image.save('/ws/external/debug_images/kitti2/sat.png')

        if 0:
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
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image0, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 5, (0, 255, 0),
                       -1)

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
            plt.savefig('/ws/external/debug_images/kitti3_s/direction.png')
            plt.show()
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
    loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’

    for _, data in zip(range(8), loader):
        print(data)


