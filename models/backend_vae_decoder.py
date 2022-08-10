import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    model = VAE_Decoder()
    