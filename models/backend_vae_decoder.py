import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class VAE_Decoder(nn.Module):
    def __init__(self,
        out_channels=3,
        latent_dim=128,
        hidden_dims: List=None
    ):
        super(VAE_Decoder, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # from a normal distribution with mean 0 and variance 1
        return eps * std + mu

    def forward(self, input):
        mu, logvar = input[0], input[1]
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z).view(-1, 512, 2, 2)
        out = self.decoder(z)
        out = self.final_layer(out)
        return [out, mu, logvar]

    def sample(self, z):
        z = self.decoder_input(z).view(-1, 512, 2, 2)
        out = self.decoder(z)
        out = self.final_layer(out)
        return out

if __name__ == '__main__':
    model = VAE_Decoder()
    mu = torch.rand(1, 128)
    logvar = torch.rand(1, 128)
    out = model([mu, logvar])
    print(out.shape)