import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class VAE_Encoder(nn.Module):
    def __init__(self, 
        in_channels=3, 
        latent_dim=128,
        hidden_dims: List=None
    ):
        super(VAE_Encoder, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # 64 ->32->16->8->4->2 (2*2)
        # This model is fixed to 64x64 input size
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        out = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)

        return [mu, log_var]


if __name__ == '__main__':
    model = VAE_Encoder()
    x = torch.rand(1, 3, 64, 64)
    out = model(x)
    print(len(out))
    print(out[0].shape)
    print(out[1].shape)
