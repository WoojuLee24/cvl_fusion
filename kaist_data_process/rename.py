import os
root = "/ws/data/Kaist_Kitti"

satmap_dir = os.path.join(root, 'satmap')
satmap_list = sorted(os.listdir(satmap_dir))

for i, satmap_file in enumerate(satmap_list):
    if '_map' in satmap_file:
        src = os.path.join(satmap_dir, satmap_file)
        dst = src[:-8] + '.png'
        os.rename(src, dst)
