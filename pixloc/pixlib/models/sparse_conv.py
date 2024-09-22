import torch
import spconv.pytorch as spconv
import torch.nn as nn

class PointCloud3DConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointCloud3DConv, self).__init__()

        # 3D Sparse Convolution Layer (spconv)
        self.sparse_conv3d = spconv.SparseConv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1,
                                                 bias=False)

        # Batch normalization and activation
        self.batch_norm3d = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, coordinates, spatial_shape, batch_size):
        """
        features: [B, N, C] -> feature matrix (B=batch size, N=number of points, C=channels)
        coordinates: [B, N, 3] -> coordinates matrix (B=batch size, N=number of points, 3=XYZ coordinates)
        spatial_shape: list of int -> the spatial dimensions (e.g., [X, Y, Z]) of the grid
        batch_size: int -> number of batches
        """
        # Reshape coordinates to [N_total, 4] where the first column is batch index
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(batch_size, coordinates.shape[1], 1)  # [B, N, 1]
        indices = torch.cat([batch_indices, coordinates], dim=-1)  # [B, N, 4] (batch index + XYZ coordinates)
        indices = indices.view(-1, 4)  # Flatten to [B*N, 4]

        # Reshape features to [N_total, C]
        features = features.view(-1, features.shape[-1])  # [B*N, C]

        # Create sparse tensor
        input_sp_tensor = spconv.SparseConvTensor(features=features, indices=indices.int(), spatial_shape=spatial_shape,
                                                  batch_size=batch_size)

        # 3D sparse convolution
        x = self.sparse_conv3d(input_sp_tensor)
        x = x.features  # Extract the dense feature tensor
        x = self.batch_norm3d(x)
        x = self.relu(x)

        # Reshape back to [B, N, C]
        output_features = x.view(batch_size, -1, x.shape[-1])  # [B, N, C]

        return output_features