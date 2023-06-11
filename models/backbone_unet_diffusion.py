"""
Code Reference:
- https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing
"""

import torch
from torch import nn
import math

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))

        # Extend last 2 dims
        time_emb = time_emb[(..., ) + (None, ) * 2]

        # Add time channel
        h = h + time_emb

        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))

        # Down or Upsample
        h = self.transform(h)

        return h
    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings
    

class SimpleUnet(nn.Module):
    """
    A simplified variant of the U-Net architecture.
    """
    def __init__(self,
        image_channels = 3,
        down_channels = [64, 128, 256, 512, 1024],
        up_channels = [1024, 512, 256, 128, 64],
        out_dim = 3,
        time_emb_dim = 32,
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                for i in range(len(down_channels) - 1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embed timestep
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()

            x = torch.cat([x, residual_x], dim=1)
            x = up(x, t)

        return self.output(x)
    

if __name__ == "__main__":
    # model = SimpleUnet()
    # print("Num params: ", sum(p.numel() for p in model.parameters()))

    # print(model(torch.randn(1, 3, 256, 256), torch.randn(1, 1)))
    # print(model)

    timestep = torch.tensor([0.5])  # Example timestep value
    input_data = torch.randn(1, 3, 256, 256)

    # Instantiate the SimpleUnet model
    model = SimpleUnet()

    # Pass the input through the model
    output = model(input_data, timestep)

    # Print the output shape
    print(output.shape)