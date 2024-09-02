# for each query image, find the nearest satellite image, and calculate their distance

#from input_libs import *
import Kitti_gps_coord_func as gps_func
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

root = "/ws/data/kitti-vo"
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'
zoom = 18


def find_query_index(idx, array, query):
    if query.shape != (1, 3):
        raise ValueError("Query array must be of shape (1, 3)")

    if array.shape[1] != 3:
        raise ValueError("Input array must have shape [N, 3]")

    if np.array_equal(array[idx], query[0]):
        return idx, array[idx]
    else:
        print(f'idx {idx} not equal')

    for idx, row in enumerate(array):
        if np.array_equal(row, query[0]):  # query[0]을 사용해 [3] 형태로 비교
            return idx, row


    raise ValueError("No query array in Input arrayy")


# satellite images gps
gps_center_file = os.path.join(root, grdimage_dir, 'satellite_gps_center2.npy')
Geodetic = np.load(gps_center_file)
NED_coords_satellite = np.zeros((Geodetic.shape[0], 3))
for i in range(Geodetic.shape[0]):
    x, y, z = gps_func.GeodeticToEcef(Geodetic[i, 0] * np.pi / 180.0, Geodetic[i, 1] * np.pi / 180.0,
                                      Geodetic[i, 2])
    xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, Geodetic[0, 0] * np.pi / 180.0,
                                            Geodetic[0, 1] * np.pi / 180.0, Geodetic[0, 2])
    NED_coords_satellite[i, 0] = xEast
    NED_coords_satellite[i, 1] = yNorth
    NED_coords_satellite[i, 2] = zUp
neigh = NearestNeighbors(n_neighbors=1)    
neigh.fit(NED_coords_satellite)

# query image gps
# read form txt files
pair = {}
idx = 0

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

            location = np.asarray([location])

            indices, sat_gps = find_query_index(idx, Geodetic, location)
            idx += 1
            # # get END of query
            # x, y, z = gps_func.GeodeticToEcef(location[0] * np.pi / 180.0, location[1] * np.pi / 180.0, location[2])
            # xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, Geodetic[0, 0] * np.pi / 180.0,
            #                                         Geodetic[0, 1] * np.pi / 180.0, Geodetic[0, 2])
            # NED_coords_query = np.array([[xEast, yNorth, zUp]])
            # _, indices = neigh.kneighbors(NED_coords_query, return_distance=True)
            # indices = indices.ravel()[0]
            #
            # # find sat image
            # sat_gps = Geodetic[indices]
            # # SatMap_name = "i" + str(indices) + "_lat_" + str(
            # #     sat_gps[0]) + "_long_" + str(
            # #     sat_gps[1]) + "_zoom_" + str(
            # #     zoom) + "_size_" + str(640) + "x" + str(640) + "_scale_" + str(2) + ".png"


            SatMap_name = "satellite_" + str(indices) + "_lat_" + str(
                sat_gps[0]) + "_long_" + str(
                sat_gps[1]) + "_zoom_" + str(
                zoom) + "_size_" + str(640) + "x" + str(640) + "_scale_" + str(2) + ".png"
            pair[line] = SatMap_name

# save the ground-view query to satellite matching pair
np.save(os.path.join(root, grdimage_dir, 'groundview_satellite_pair2.npy') , pair)








