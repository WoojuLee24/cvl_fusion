import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors
import os.path
import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import transformations
import yaml
import math

root_folder = "/ws/data/Ford_AV/"
log_id = "2017-08-04-V2-Log4(2)" #"2017-10-26-V2-Log1"

log_list = [
            # '2017-07-24-V2-Log1',
            #     '2017-07-24-V2-Log2',
            #     '2017-07-24-V2-Log3',
            #     '2017-07-24-V2-Log4',
            #     '2017-07-24-V2-Log5',
            #     '2017-07-24-V2-Log6',
                '2017-08-04-V2-Log1',
                # '2017-08-04-V2-Log2',
                # '2017-08-04-V2-Log3',
                # '2017-08-04-V2-Log4',
                # '2017-08-04-V2-Log5',
                # '2017-08-04-V2-Log6',
                '2017-10-26-V2-Log1',
                # '2017-10-26-V2-Log2',
                # '2017-10-26-V2-Log3',
                # '2017-10-26-V2-Log4',
                '2017-10-26-V2-Log5',
                # '2017-10-26-V2-Log6',

                ]

for log_id in log_list:
    log_folder = os.path.join(root_folder, log_id)

    if not os.path.exists(log_folder):
        print(f"{log_id} does not exist.")

    # for dir in os.listdir(log_folder):
        # log-FL/RR~
        # subdir = os.path.join(log_folder, dir)
        # if not os.path.isdir(subdir):
        #     continue

    subdir = os.path.join(log_folder, log_id +'-FL')
    # if ('-FL' in dir) or ('-RR' in dir) or ('-SL' in dir) or ('-SR' in dir):
    #     print('process '+dir)
    # else:
    #     continue

    if not os.path.exists(subdir):
        continue

    file_list = os.listdir(subdir)
    file_list.sort()

    # # ignore reconstruction images
    # if '2017-10-26-V2-Log1' in subdir:
    #     file_list = file_list[:9660]+file_list[11261:]
    # if '2017-08-04-V2-Log1' in subdir:
    #     file_list = file_list[:8330]+file_list[9730:]

    os.makedirs(os.path.join(log_folder, 'info_files'), exist_ok=True)
    txt_file_name = os.path.join(log_folder, 'info_files', log_id + '-FL' + '-names.txt')

    with open(txt_file_name, 'w') as f:
        f.write(str(dir) + '\n')
        for name in file_list:
            f.write(str(name)+'\n')