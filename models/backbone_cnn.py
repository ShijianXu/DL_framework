import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,
        input_channels,
        output_channels,
        activation='relu'
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=output_channels, kernel_size=5)
        self.activ = getattr(F, activation)

    def forward(self, x):
        x = self.pool(self.activ(self.conv1(x)))
        out = self.pool(self.activ(self.conv2(x)))
        return out

if __name__ == '__main__':
    model = SimpleCNN(3, 16)
    x = torch.rand(4, 3, 32, 32)
    out = model(x)
    print(out.shape)