# given the center geodestic coordinate of each satellite patch
# retrieve satellite patchs from the google map server

# Todo: using the viewing direction of forward camera to move the center of satellite patch
# without this, the satellite patch only share a small common FoV as the ground-view query image

# NOTE:
# You need to provide a key
# keys = ['**your key**']
keys = [
'AIzaSyCk0SAwBKvvZCK0Ql778PnB4SM8aFySNqk'#'AIzaSyDNBScfkqTlt59jeDGsVXlVNSvDpkiKrRs'
]

import requests
from io import BytesIO
import os
import time
from PIL import Image as PILI
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# root_dir = "/ws/data/kitti-vo"
# satmap_dir = 'satmap_18'
# grdimage_dir = 'raw_data'
root_dir = "/ws/data/kitti-vo"
satmap_dir = 'satmap_z18_s2'
grdimage_dir = 'raw_data'
gps_center = 'satellite_gps_center2.npy' # 'satellite_gps_center.npy'

with open(os.path.join(root_dir, grdimage_dir, gps_center), 'rb') as f:
    Geodetic = np.load(f)


url_head = 'https://maps.googleapis.com/maps/api/staticmap?'
zoom = 18
sat_size = [640, 640]
maptype = 'satellite'
scale = 2

nb_keys = len(keys)

nb_satellites = Geodetic.shape[0]

satellite_folder = os.path.join(root_dir, satmap_dir)

if not os.path.exists(satellite_folder):
    os.makedirs(satellite_folder)

# for i in tqdm(range(nb_satellites)):
#
#     lat_a, long_a, height_a = Geodetic[i, 0], Geodetic[i, 1], Geodetic[i, 2]
#
#     image_name = satellite_folder + "/satellite_" + str(i) + "_lat_" + str(lat_a) + "_long_" + str(
#         long_a) + "_zoom_" + str(
#         zoom) + "_size_" + str(sat_size[0]) + "x" + str(sat_size[0]) + "_scale_" + str(scale) + ".png"
#
#     if os.path.exists(image_name):
#         continue
#
#     time.sleep(1)
#
#     saturl = url_head + 'center=' + str(lat_a) + ',' + str(long_a) + '&zoom=' + str(
#         zoom) + '&size=' + str(
#         sat_size[0]) + 'x' + str(sat_size[1]) + '&maptype=' + maptype + '&scale=' + str(
#         scale) + '&format=png32' + '&key=' + \
#              keys[0]
#     #f = requests.get(saturl, stream=True)
#
#     try:
#         f = requests.get(saturl, stream=True)
#         f.raise_for_status()
#     except requests.exceptions.HTTPError as err:
#         raise SystemExit(err)
#
#     bytesio = BytesIO(f.content)
#     cur_image = PILI.open(bytesio)
#
#     cur_image.save(image_name)



def download_image(i):
    lat_a, long_a, height_a = Geodetic[i, 0], Geodetic[i, 1], Geodetic[i, 2]

    image_name = satellite_folder + "/satellite_" + str(i) + "_lat_" + str(lat_a) + "_long_" + str(
        long_a) + "_zoom_" + str(
        zoom) + "_size_" + str(sat_size[0]) + "x" + str(sat_size[0]) + "_scale_" + str(scale) + ".png"

    if os.path.exists(image_name):
        return

    time.sleep(1)

    saturl = url_head + 'center=' + str(lat_a) + ',' + str(long_a) + '&zoom=' + str(
        zoom) + '&size=' + str(
        sat_size[0]) + 'x' + str(sat_size[1]) + '&maptype=' + maptype + '&scale=' + str(
        scale) + '&format=png32' + '&key=' + \
             keys[0]

    try:
        response = requests.get(saturl, stream=True)
        response.raise_for_status()
        bytesio = BytesIO(response.content)
        cur_image = PILI.open(bytesio)
        cur_image.save(image_name)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {i}: {e}")


max_workers = 1
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_image, i) for i in range(nb_satellites)]
    for future in tqdm(as_completed(futures), total=len(futures)):
        pass