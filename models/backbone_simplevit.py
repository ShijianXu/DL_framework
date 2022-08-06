# source: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from utils import pair

class SimpleViT(nn.Module):
    def __init__(self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        

    def forward(self, x):
        pass