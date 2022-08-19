# https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py
# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import pair

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MixerLayer(nn.Module):
    def __init__(self, num_patches, dim, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MLP(num_patches, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, channel_dim, dropout)
        )

    def forward(self, x):
        # x: N x dim
        x = self.token_mix(x) + x
        x = self.channel_mix(x) + x
        return x


class MLPMixer(nn.Module):
    def __init__(self,
        *, 
        image_size, 
        patch_size, 
        dim, 
        depth, 
        num_classes, 
        token_dim = 256, 
        channel_dim = 2048,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.mixer = nn.ModuleList([])
        for _ in range(depth):
            self.mixer.append(
                MixerLayer(num_patches, dim, token_dim, channel_dim, dropout)
            )

        self.layer_norm = nn.LayerNorm(dim)
        
        self.linear_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)    # B x N x C

        for layer in self.mixer:
            x = layer(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)           # B x C
        x = self.linear_head(x)
        return x


if __name__ == '__main__':
    model = MLPMixer(
        image_size=224,
        channels=3,
        patch_size=16,
        dim=512,
        depth=8,
        num_classes=1000,
        token_dim=256,
        channel_dim=2048
    )

    import numpy as np
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)