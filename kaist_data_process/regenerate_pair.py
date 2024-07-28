import numpy as np
import os

path = '/ws/data/Kaist_Kitti_Dense/raw_data/groundview_satellite_pair.npy'
pair_npy = np.load(path, allow_pickle=True)
pairs = pair_npy.item()

i = 0
iters = 20
new_path = f'/ws/data/Kaist_Kitti_Dense/raw_data/groundview_satellite_pair_{iters}iters.npy'
new_pairs = {}
for query, ref in pairs.items():
    if i % iters == 0:
        key_ref = ref
    new_pairs[query] = key_ref
    i+=1

np.save(new_path, new_pairs)
new_pairs_npy = np.load(new_path, allow_pickle=True)
new_pairs_loaded = new_pairs_npy.item()
pass

