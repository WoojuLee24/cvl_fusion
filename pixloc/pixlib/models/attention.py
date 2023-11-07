import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pixloc.pixlib.datasets.kitti as kitti

# key_points_2d = torch.from_numpy(camera_k).float() @ key_points.T
# key_points_2d = key_points_2d[:2, :] / key_points_2d[2, :]
# key_points_2d = key_points_2d.T
# grd_image['points2D'] = key_points_2d

grd_process_size = kitti.grd_process_size

class Attention_Module(nn.Module):
    def __init__(self, num_points):
        super(Attention_Module, self).__init__()

        self.fc = nn.Linear(2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.attention_weights = nn.Parameter(torch.ones(num_points, 1))

    def forward(self, lidar_points, original_img):
        # lidar_points: 2D lidar points (u, v) - shape: (num_points, 2)
        # original_img: shape:(batch_size, 3, height, width)
        img_height = original_img.shape[2]
        img_width = original_img.shape[3]

        # compute attention weights
        attention_input = self.fc(lidar_points)
        attention_weights = self.sigmoid(attention_input)
        attention_weights = attention_weights * self.attention_weights

        # Normalize attention weights
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)

        # Stack

        # Bilinear interpolation
        lidar_points_int = lidar_points.floor().long()
        lidar_points_frac = lidar_points - lidar_points_int.float()

        top_left = (1 - lidar_points_frac[:, 0]) * (1 - lidar_points_frac[:, 1])
        top_right = lidar_points_frac[:, 0] * (1 - lidar_points_frac[:, 1])
        bottom_left = (1 - lidar_points_frac[:, 0]) * lidar_points_frac[:, 1]
        bottom_right = lidar_points_frac[:, 0] * lidar_points_frac[:, 1]

        # attended image


        return attention_weights, attended_img



'''
def lidar_mask(data):
    lidar_points = data['query']['points2D'].squeeze()  # (1024, 2)
    original_img = data['query']['image'].squeeze()  # (3, 384, 1248)
    original_img = original_img.float()/255.0
    img_height = grd_process_size[0]  # 384
    img_width = grd_process_size[1]  # 1248

    lidar_points_int = lidar_points.floor().long()
    lidar_points_frac = lidar_points - lidar_points_int.float()

    # Bilinear interpolation
    top_left = (1 - lidar_points_frac[:, 0]) * (1 - lidar_points_frac[:, 1])
    top_right = lidar_points_frac[:, 0] * (1 - lidar_points_frac[:, 1])
    bottom_left = (1 - lidar_points_frac[:, 0]) * lidar_points_frac[:, 1]
    bottom_right = lidar_points_frac[:, 0] * lidar_points_frac[:, 1]

    lidar_mask = torch.zeros((img_height, img_width))

    for i in range(lidar_points.shape[0]):
        u, v = lidar_points_int[i]
        if 0 <= u < img_width - 1 and 0 <= v < img_height - 1:
            lidar_mask[v, u] += top_left[i]
            lidar_mask[v, u + 1] += top_right[i]
            lidar_mask[v + 1, u] += bottom_left[i]
            lidar_mask[v + 1, u + 1] += bottom_right[i]

    masked_img = original_img * lidar_mask

    # data['query']['attention_mask'] = attention_mask
    # data['query']['attended_img'] = attended_img

    return lidar_mask
'''

# for visualization
if __name__ == '__main__':
    SavePlt = True
    Test_img = 5

    # test to load 1 data
    conf = {
        'max_num_points3D': 1024,
        'force_num_points3D': True,
        'batch_size': 1,
        'seed': 1,
        'num_workers': 0,
    }
    dataset = kitti.Kitti(conf)
    loader = dataset.get_data_loader('train', shuffle=True)

    for i, data in zip(range(Test_img), loader):
        # attention module
        data = attention_module(data)
        attention_mask = data['query']['attention_mask']
        original_img = data['query']['image'].squeeze()
        lidar_points = data['query']['points2D'].squeeze()

        # visualize
        plt.subplot(1, 3, 1)
        plt.imshow(attention_mask)
        plt.title('Attention Mask')

        plt.subplot(1, 3, 2)
        plt.imshow(original_img.permute(1, 2, 0))
        plt.title('Original Image')

        plt.subplot(1, 3, 3)
        plt.imshow(data['query']['attended_img'].permute(1, 2, 0))
        plt.title('Attended Image')

        if SavePlt:
            save_path = '/ws/external/visualization/attention'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'attention_'+str(i)+'.png'))