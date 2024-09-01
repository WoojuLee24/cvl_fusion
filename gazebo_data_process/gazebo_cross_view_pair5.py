import os
import numpy as np
import random

root = "/ws/data/gazebo_kitti"
grd_root = os.path.join(root, 'raw_data')
sat_root= os.path.join(root, 'satmap')
grd_image_dir_list = ['2024_08_12/lakepark1', '2024_08_12/lakepark2', '2024_08_12/lakepark3']
sat_image_dir_list = ['2024_08_12/lakepark1', '2024_08_12/lakepark2', '2024_08_12/lakepark3']
random_sat = ['fake1', 'fake2', 'fake3', 'fake4', 'gt']   # to noisy!
# random_sat = ['gt']
image_dir = 'image_02/data'
oxts_dir = 'oxts/data'
zoom = 18

pair = {}

for i, sat_log in enumerate(sat_image_dir_list):
    grd_log = grd_image_dir_list[i]
    sat_real_dir = os.path.join(sat_root, sat_log)
    grd_real_dir = os.path.join(grd_root, grd_log, image_dir)
    grdimage_list = sorted(os.listdir(grd_real_dir))
    for j, grd_file in enumerate(grdimage_list):
        grd_record_file = os.path.join(grd_log, grd_file)

        random_dir = random.choice(random_sat)
        sat_random_dir = os.path.join(sat_log, random_dir)
        satmap_list = sorted(os.listdir(os.path.join(sat_real_dir, random_dir)))
        satmap_record_file = os.path.join(sat_random_dir, satmap_list[j])

        pair[grd_record_file] = satmap_record_file

np.save(os.path.join(root, 'raw_data/groundview_satellite_pair5.npy'), pair)





