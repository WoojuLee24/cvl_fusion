# from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)    # [b, hw, c]

        qkv = self.to_qkv(x).chunk(3, dim = -1) # [b, hw, c] -> [b, hw, c] * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # group. multi-head attention # [b, g, hw, c//g]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    #  [b, g, hw, hw]

        attn = self.attend(dots)    # [b, g, hw, hw]

        out = torch.matmul(attn, v) #  # [b, g, hw, hw] @ [b, g, hw, c//g] -> [b, g, hw, c//g]
        out = rearrange(out, 'b h n d -> b n (h d)')    # [b, hw, c]
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim1, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim2, inner_dim * 2, bias = False)

        self.to_out = nn.Linear(inner_dim, dim1, bias = False)

    def forward(self, x, y):
        x = self.norm1(x)    # [b, hw, c]
        y = self.norm2(y)

        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim = -1) # [b, hw, c] -> [b, hw, c] * 3
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)  # group. multi-head attention # [b, g, hw, c//g]
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    #  [b, g, hw, hw]

        attn = self.attend(dots)    # [b, g, hw, hw]

        out = torch.matmul(attn, v) #  # [b, g, hw, hw] @ [b, g, hw, c//g] -> [b, g, hw, c//g]
        out = rearrange(out, 'b h n d -> b n (h d)')    # [b, hw, c]
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x # [b, hw, c]
            x = ff(x) + x   # [b, hw, c]
        return x


class CrossTransformer(nn.Module):
    def __init__(self, dim1, dim2, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim1, dim2, heads = heads, dim_head = dim_head),
                FeedForward(dim1, mlp_dim)
            ]))
    def forward(self, x, y):    # x: query, y: ref
        for attn, ff in self.layers:
            x = attn(x, y) + x # [b, hw, c]
            x = ff(x) + x   # [b, hw, c]
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)    # [b, 3, h*p1 ,w*p2] -> [b, h, w, p1*p2*3] -> [b, h, w, c]
        pe = posemb_sincos_2d(x)    # [h*w, c]
        x = rearrange(x, 'b ... d -> b (...) d') + pe   # [b, h*w, c]

        x = self.transformer(x) # [b, hw, c] -> [b, hw, c]
        x = x.mean(dim = 1) # [b, c]

        x = self.to_latent(x)
        return self.linear_head(x)  # [b, classes]
