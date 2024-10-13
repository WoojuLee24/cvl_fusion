import torch
import torch.nn as nn
import torch.nn.functional as F
from pixloc.pixlib.geometry.interpolation import Interpolator


class DenseEncoder(nn.Module):
    def __init__(self, cout, normalize='l2'):
        super(DenseEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, cout[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cout[0], cout[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cout[1], cout[0], kernel_size=3, stride=1, padding=1),
        )
        self.interpolator = Interpolator()
        self.normalize = normalize

    def forward(self, x, p2D):
        x = self.net(x)
        y, mask, _ = self.interpolator(x, p2D)
        if self.normalize:
            y = F.normalize(y, dim=-1)
        return y

