import os
root = "/ws/data/Kaist_Kitti_Dense" # "/ws/data/kaist_mobile"

satmap_dir = os.path.join(root, 'satmap')
satmap_list = sorted(os.listdir(satmap_dir))
grd_dir = os.path.join(root, 'raw_data/image/data')
grd_list = sorted(os.listdir(grd_dir))

train_list = []
val_list = []
test_list = []
for i, grd_file in enumerate(grd_list):
    test_list.append(grd_file)
    # if int(grd_file[:-4]) > 1390 and int(grd_file[:-4]) < 1600:
    #     test_list.append(grd_file)
    # elif int(grd_file[:-4]) % 50 != 0:
    #     train_list.append(grd_file)
    # elif int(grd_file[:-4]) % 50 == 0:
    #     val_list.append(grd_file)

train_file = os.path.join(root, 'raw_data/split', 'train_files.txt')
val_file = os.path.join(root, 'raw_data/split', 'val_files.txt')
test_file = os.path.join(root, 'raw_data/split', 'test_files.txt')

with open(train_file, 'w') as file:
    file.write('\n'.join(train_list))
with open(val_file, 'w') as file:
    file.write('\n'.join(val_list))
with open(test_file, 'w') as file:
    file.write('\n'.join(test_list))

