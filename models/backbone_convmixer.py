# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convmixer.py

import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self,
        dim, 
        depth, 
        kernel_size=9, 
        patch_size=7, 
        in_chans=3, 
        num_classes=1000, 
        activation='gelu', 
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = dim
        
        if activation == 'gelu':
            self.activation = nn.GELU

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            self.activation(),
            nn.BatchNorm2d(dim)
        )       # To patch embedding: BxCxHxW -> Bxdx(H//p)x(W//p)
        
        self.mixer = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        self.activation(),
                        nn.BatchNorm2d(dim)
                    )),     # spatial mixer
                    nn.Conv2d(dim, dim, kernel_size=1),     # channel mixer
                    self.activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.mixer(x)
        x = self.pooling(x)
        return x

    def forward(self, img):
        x = self.forward_features(img)
        x = self.head(x)

        return x


if __name__ == '__main__':
    model = ConvMixer(
        dim=128, 
        depth=6, 
        kernel_size=9, 
        patch_size=7, 
        in_chans=3, 
        num_classes=10, 
    )

    print(model)

    x = torch.rand(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
