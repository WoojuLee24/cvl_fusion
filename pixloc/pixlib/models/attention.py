import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pixloc.pixlib.datasets.kitti as kitti
import pixloc.pixlib.geometry.costs as costs

# key_points_2d = torch.from_numpy(camera_k).float() @ key_points.T
# key_points_2d = key_points_2d[:2, :] / key_points_2d[2, :]
# key_points_2d = key_points_2d.T
# grd_image['points2D'] = key_points_2d

grd_process_size = kitti.grd_process_size
'''
class DirectAbsoluteCost:
    def __init__(self, interpolator: Interpolator, normalize: bool = True):
        self.interpolator = interpolator
        self.normalize = normalize
    def residuals(
            self, T_q2r: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor, int]] = None,
            do_gradients: bool = False):

        p3D_r = T_q2r * p3D # q_3d to q2r_3d
        p2D, visible = camera.world2image(p3D_r) # q2r_3d to q2r_2d
        F_p2D_raw, valid, gradients = self.interpolator(
            F_ref, p2D, return_gradients=do_gradients) # get g2r 2d features
        valid = valid & visible

        C_ref, C_query, C_count = confidences

        C_ref_p2D, _, _ = self.interpolator(C_ref, p2D, return_gradients=False) # get ref 2d confidence

        # the first confidence
        weight = C_ref_p2D[:, :, 0] * C_query[:, :, 0]
        if C_count > 1:
            grd_weight = C_ref_p2D[:, :, 1].detach() * C_query[:, :, 1]
            weight = weight * grd_weight
        # if C2_start == 0:
        #     # only grd confidence:
        #     # do not gradiant back to ref confidence
        #     weight = C_ref_p2D[:, :, 0].detach() * C_query[:, :, 0]
        # else:
        #     weight = C_ref_p2D[:,:,0] * C_query[:,:,0]
        # # the second confidence
        # if C_query.shape[-1] > 1:
        #     grd_weight = C_ref_p2D[:, :, 1].detach() * C_query[:, :, 1]
        #     grd_weight = torch.cat([torch.ones_like(grd_weight[:, :C2_start]), grd_weight[:, C2_start:]], dim=1)
        #     weight = weight * grd_weight

        if weight != None:
            weight = weight.masked_fill(~(valid), 0.)
            #weight = torch.nn.functional.normalize(weight, p=float('inf'), dim=1) #??

        if self.normalize: # huge memory
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_query
        info = (p3D_r, F_p2D, gradients) # ref information
        return res, valid, weight, F_p2D, info
'''
class Attention_Module(nn.Module):
    def __init__(self, num_points):
        super(Attention_Module, self).__init__()

        self.fc = nn.Linear(2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.attention_weights = nn.Parameter(torch.ones(num_points, 1))

    def forward(self, lidar_points):
        # lidar_points: 2D lidar points (u, v) - shape: (num_points, 2)

        # Attention weights
        attention_weights = self.fc(lidar_points)
        attention_weights = attention_weights * self.attention_weights
        attention_weights = self.sigmoid(attention_weights)
        attention_weights.squeeze_()

        return attention_weights

    def img_attention(self, lidar_points, original_img, attention_weights):
        # original_img: shape:(3, height, width)
        img_height = original_img.shape[1]
        img_width = original_img.shape[2]

        # Bilinear interpolation
        lidar_points_int = lidar_points.floor().long()
        lidar_points_frac = lidar_points - lidar_points_int.float()

        top_left = (1 - lidar_points_frac[:, 0]) * (1 - lidar_points_frac[:, 1])
        top_right = lidar_points_frac[:, 0] * (1 - lidar_points_frac[:, 1])
        bottom_left = (1 - lidar_points_frac[:, 0]) * lidar_points_frac[:, 1]
        bottom_right = lidar_points_frac[:, 0] * lidar_points_frac[:, 1]

        top_left = top_left * attention_weights
        top_right = top_right * attention_weights
        bottom_left = bottom_left * attention_weights
        bottom_right = bottom_right * attention_weights

        attention_mask = torch.zeros((img_height, img_width))
        for i in range(lidar_points.shape[0]):
            u, v = lidar_points_int[i]
            if 0 <= u < img_width - 1 and 0 <= v < img_height - 1:
                attention_mask[v, u] += top_left[i]
                attention_mask[v, u + 1] += top_right[i]
                attention_mask[v + 1, u] += bottom_left[i]
                attention_mask[v + 1, u + 1] += bottom_right[i]

        # attended image
        attended_img = original_img * attention_mask

        return attention_mask, attended_img

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
    num_points = conf["max_num_points3D"]
    attention_module = Attention_Module(num_points)
    dataset = kitti.Kitti(conf)
    loader = dataset.get_data_loader('train', shuffle=True)

    for i, data in zip(range(Test_img), loader):
        # attention module
        lidar_points = data['query']['points2D'].squeeze()
        original_img = data['query']['image'].squeeze()

        attention_weights = attention_module.forward(lidar_points)
        attention_mask, attended_img = attention_module.img_attention(lidar_points, original_img, attention_weights)

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