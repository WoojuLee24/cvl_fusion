import os
import numpy as np

root = "/ws/data/Kaist_Kitti"
grdimage_dir = 'raw_data/image/data'
oxts_dir = 'oxts/data'
zoom = 18

satmap_dir = os.path.join(root, 'satmap')
grdimage_dir = os.path.join(root, grdimage_dir)

satmap_list = sorted(os.listdir(satmap_dir))
grdimage_list = sorted(os.listdir(grdimage_dir))

pair = {}

for i, satmap_file in enumerate(satmap_list):
    grdimage_name = grdimage_list[i]
    grdimage_file = 'image/data/' + grdimage_name
    pair[grdimage_file] = satmap_file

np.save(os.path.join(root, 'raw_data/groundview_satellite_pair.npy'), pair)





