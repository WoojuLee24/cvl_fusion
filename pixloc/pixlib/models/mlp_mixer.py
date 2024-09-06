import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MLPMixer(nn.Module):
    def __init__(self, mode='mixer', in_channels=3, num_patches=256,
                 hidden_size=512, hidden_s=256, hidden_c=2048,
                 num_layers=8, num_classes=3, drop_p=0., off_act=False, is_cls_token=False):

        super(MLPMixer, self).__init__()
        # num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        # self.patch_emb = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b d h w -> b (h w) d')
        # )
        self.conv = nn.Linear(in_channels, hidden_size)

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(mode, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act)
            for _ in range(num_layers)
            ]
        )
        # self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # out = self.patch_emb(x)
        out = self.conv(x)
        # if self.is_cls_token:
        #     out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        # out = self.ln(out)
        # out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        # out = self.clf(out)
        return out


class MixerLayer(nn.Module):
    def __init__(self, mode, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mode = mode
        if self.mode == 'mixer':
            self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
            self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
        elif self.mode == 'mixer_c':
            self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
        elif self.mode == 'mixer_s':
            self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)

    def forward(self, x):
        if self.mode == 'mixer':
            out = self.mlp1(x)
            out = self.mlp2(out)
        elif self.mode == 'mixer_c':
            out = self.mlp2(x)
        elif self.mode == 'mixer_s':
            out = self.mlp1(x)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        # self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.relu if not off_act else lambda x:x
    def forward(self, x):
        # out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do1(self.act(self.fc1(x)))
        out = self.do2(self.fc2(out))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        # self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.relu if not off_act else lambda x:x
    def forward(self, x):
        # out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do1(self.act(self.fc1(x)))
        out = self.do2(self.fc2(out))
        return out+x