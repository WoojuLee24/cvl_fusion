import os
root = "/ws/data/gazebo_kitti"

satmap_dir = os.path.join(root, 'satmap')
satmap_list = sorted(os.listdir(satmap_dir))
grd_log_list = ['2024_07_02/lakepark1', '2024_07_02/lakepark2', '2024_07_02/lakepark3']


train_list = []
val_list = []
for grd_log in grd_log_list[:-1]:
    grd_dir = os.path.join(root, 'raw_data', grd_log, 'image_02/data')
    grd_list = sorted(os.listdir(grd_dir))

    for i, grd_file in enumerate(grd_list):
        grd_record_file = os.path.join(grd_log, grd_file)
        if int(grd_file[:-4]) % 10 != 0:
            train_list.append(grd_record_file)
        elif int(grd_file[:-4]) % 10 == 0:
            val_list.append(grd_record_file)


test_list = []
for grd_log in grd_log_list[-1:]:
    grd_dir = os.path.join(root, 'raw_data', grd_log, 'image_02/data')
    grd_list = sorted(os.listdir(grd_dir))
    for i, grd_file in enumerate(grd_list):
        grd_record_file = os.path.join(grd_log, grd_file)
        test_list.append(grd_record_file)

train_file = os.path.join(root, 'raw_data/split', 'train_files.txt')
val_file = os.path.join(root, 'raw_data/split', 'val_files.txt')
test_file = os.path.join(root, 'raw_data/split', 'test_files.txt')

with open(train_file, 'w') as file:
    file.write('\n'.join(train_list))
with open(val_file, 'w') as file:
    file.write('\n'.join(val_list))
with open(test_file, 'w') as file:
    file.write('\n'.join(test_list))

