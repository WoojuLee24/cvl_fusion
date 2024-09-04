import numpy as np
import os

root = "/ws/data/kitti-vo"
satmap_dir = 'satmap_18'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'
zoom = 18


locations = []
for split in ('train','val','test'):
    txt_file_name = os.path.join(root, grdimage_dir, 'kitti_split', split + '_files.txt')
    with open(txt_file_name, "r") as txt_f:
        lines = txt_f.readlines()
        for line in lines:
            line = line.strip()
            # get location of query
            drive_dir = line[:37]
            image_no = line[38:].lower()
            oxts_file_name = os.path.join(root, grdimage_dir, drive_dir, oxts_dir,
                                          image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
            location = [float(content[0]), float(content[1]), float(content[2])]
            locations.append(location)

locations = np.asarray(locations)
np.save('/ws/data/kitti-vo/raw_data/satellite_gps_center2.npy', locations)
