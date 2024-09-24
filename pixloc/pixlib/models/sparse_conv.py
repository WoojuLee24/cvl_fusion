import torch
import spconv.pytorch as spconv
import torch.nn as nn

class SparseNet(nn.Module):
    def __init__(self, input_channels, output_channels, mode='smpconv', max_num_features=5000, stride=[1, 1]):
        super(SparseNet, self).__init__()

        self.max_num_features = max_num_features
        # # 3D Sparse Convolution Layer (spconv)
        # self.sparse_conv3d = spconv.SparseConv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1,
        #                                          bias=False)
        # self.subm_conv3d = spconv.SubMConv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1,
        #                                          bias=False)
        #
        # # Batch normalization and activation
        # self.batch_norm3d = nn.BatchNorm1d(output_channels)
        # self.relu = nn.ReLU(inplace=True)
        if mode in ['smpconv', 'smpconv1.1']:      # no BN
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False),
                # nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1,
                              bias=False),
                nn.ReLU(inplace=True),

                spconv.SparseConv3d(output_channels, output_channels, kernel_size=3, stride=stride[0], padding=1,
                                    bias=False),
                nn.ReLU(inplace=True),
                spconv.SparseConv3d(output_channels, output_channels, kernel_size=3, stride=stride[1], padding=1,
                                    bias=False),
                # spconv.ToDense(),
            )


    def forward(self, features, coordinates, spatial_shape, batch_size):
        """
        features: [B, N, C] -> feature matrix (B=batch size, N=number of points, C=channels)
        coordinates: [B, N, 3] -> coordinates matrix (B=batch size, N=number of points, 3=XYZ coordinates)
        spatial_shape: list of int -> the spatial dimensions (e.g., [X, Y, Z]) of the grid
        batch_size: int -> number of batches
        """
        # Reshape coordinates to [N_total, 4] where the first column is batch index
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(batch_size, coordinates.shape[1], 1).to(coordinates.device)  # [B, N, 1]
        indices = torch.cat([batch_indices, coordinates], dim=-1)  # [B, N, 4] (batch index + XYZ coordinates)
        indices = indices.view(-1, 4)  # Flatten to [B*N, 4]

        # Reshape features to [N_total, C]
        features = features.view(-1, features.shape[-1])  # [B*N, C]

        # Create sparse tensor
        input_sp_tensor = spconv.SparseConvTensor(features=features, indices=indices.int(), spatial_shape=spatial_shape,
                                                  batch_size=batch_size)

        x = self.net(input_sp_tensor)
        # y = self.reconstruct_batch_features(x, batch_size)
        y = self.reconstruct_batch_features_with_max_points(x, batch_size, self.max_num_features)

        return y


    def reconstruct_batch_features(self, spconv_output, B):
        indices = spconv_output.indices  # [N, 4]
        features = spconv_output.features  # [N, C]
        batch_indices = indices[:, 0]  # [N]
        point_counts = [(batch_indices == b).sum().item() for b in range(B)]
        max_points = max(point_counts)
        C = features.shape[1]
        output = torch.zeros(B, max_points, C).to(features.device)

        for b in range(B):
            idx = (batch_indices == b).nonzero(as_tuple=True)[0]
            num_points = idx.shape[0]
            output[b, :num_points, :] = features[idx, :]

        return output  # 크기: [B, max_points, C]

    def reconstruct_batch_features_with_max_points(self, spconv_output, B, max_points):
        indices = spconv_output.indices  # [N, 4]
        features = spconv_output.features  # [N, C]
        batch_indices = indices[:, 0]  # [N]
        C = features.shape[1]

        output = torch.zeros(B, max_points, C).to(features.device)

        for b in range(B):
            idx = (batch_indices == b).nonzero(as_tuple=True)[0]
            num_points = idx.shape[0]

            if num_points >= max_points:
                # 포인트가 max_points보다 많으면, 앞에서부터 max_points개를 선택합니다.
                selected_idx = idx[:max_points]
                output[b, :, :] = features[selected_idx, :]
            else:
                # 포인트가 max_points보다 적으면, 전부 복사하고 나머지는 0으로 남겨둡니다.
                output[b, :num_points, :] = features[idx, :]

        return output  # 크기: [B, max_points, C]

    def visualize_voxels(self, indices):
        from matplotlib import pyplot as plt

        voxel_coords = indices[:, 1:4].cpu().numpy()    # only coordinates

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], c='b', marker='o')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.imsave('/ws/external/debug_images/voxel.png')
        plt.show()
