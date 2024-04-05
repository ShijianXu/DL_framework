import torch
import torch.nn as nn
import torch.nn.functional as F
from module_coupling_layer import CouplingLayer
from typing import List

class NormalizingFlow(nn.Module):
    def __init__(self, 
        input_dim, 
        latent_dim,
        num_layers,
        num_masks,
        in_channels,
        num_coupling_layers,
    ):
        super(NormalizingFlow, self).__init__()

        self.coupling_layers = nn.ModuleList([
            CouplingLayer(in_channels, in_channels, mask_type='checkerboard' if i % 2 == 0 else 'channel') 
             for i in range(num_coupling_layers)
            ])

    def forward(self, x):
        log_det_J = 0
        for i in range(self.num_coupling_layers):
            x, log_det = self.coupling_layers[i](x, log_det)
            log_det_J += log_det

        return x, log_det_J

    def reverse(self, y):
        # Reverse the flow for sampling/generating images
        for i in reversed(range(self.num_coupling_layers)):
            y = self.coupling_layers[i].reverse(y)
        return y

if __name__ == '__main__':
    model = NormalizingFlow()
    x = torch.rand(1, 3, 64, 64)
    out = model(x)
    print(len(out))
    print(out[0].shape)
    print(out[1].shape)