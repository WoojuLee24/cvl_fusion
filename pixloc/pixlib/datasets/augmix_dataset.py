# written by jay
# refer to https://github.com/google-research/augmix
# ===================================================

import torch
from typing import Dict, List, Optional, Tuple
import torchvision.transforms.functional as F
from pixloc.pixlib.datasets.augmix import AugMix

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(
        self,
        args,
        dataset,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None
    ) -> None:
        self.args = args
        self.dataset = dataset

        self.severity = int(severity)
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.interpolation = interpolation
        self.fill = fill
        self.augmix = AugMix(self.severity, self.mixture_width, self.chain_depth, self.alpha, self.interpolation, self.fill)

    def __getitem__(self, i):

        data = self.dataset[i]

        if self.args.aug == 'augmix1':
            data['ref']['image'] = self.augmix(data['ref']['image'])
            data['query']['image'] = self.augmix(data['query']['image'])
        elif self.args.aug == 'augmix1.g':
            data['query']['image'] = self.augmix(data['query']['image'])

        return data

    def __len__(self):
        return len(self.dataset)