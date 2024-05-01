import torch
import torch.nn as nn
from typing import *

def create_checkerboard_mask(h, w, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(c_in, invert=False):
    mask = torch.zeros(c_in, dtype=torch.float32)
    mask[::c_in // 2] = 1
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.
    """
    def __init__(self, net, in_channels, mask) -> None:
        """
        Args:
            net: nn.Module, a neural network that predicts the scale and shift parameters
            mask: torch.Tensor, a 1D tensor of 0s and 1s that determines which elements of the input are transformed
            in_channels: int, the number of channels in the input
        """
        super().__init__()
        self.net = net
        self.in_channels = in_channels
        self.register_buffer('mask', mask)

    def forward(self, x, log_det_J, orig_x=None):
        """
        orig_x is only needed in VarDeq.
        """
        x1 = x * self.mask
        x2 = x * (1 - self.mask)

        # apply network to masked input
        if orig_x is not None:
            nn_out = self.net(torch.concat([x1, orig_x], dim=1))
        else:
            nn_out = self.net(x1)

        log_scale, shift = nn_out.chunk(2, dim=1)   # split along channel dimension into the scale and shift parts
        scale = torch.exp(log_scale)                # scale must be positive, so we exponentiate

        y2 = x2*scale + shift
        y = y2 * (1-self.mask) + x1

        log_det_J += torch.sum(log_scale, dim=(1, 2, 3))

        return y, log_det_J

    def reverse(self, y, log_det_J):
        y1 = y * self.mask
        y2 = y * (1 - self.mask)

        nn_out = self.net(y1)
        log_scale, shift = nn_out.chunk(2, dim=1)
        scale = torch.exp(log_scale)

        x2 = (y2 - shift) / scale
        x = x2 * (1 - self.mask) + y1

        log_det_J -= torch.sum(log_scale, dim=(1, 2, 3))

        return x, log_det_J


#===================================================================

class IrisCouplingLayer(nn.Module):
    def __init__(
        self, 
        net: nn.Module, 
        split: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.net = net
        self.split = split

    def forward(self, x, log_det_J):
        """
        log_det_J: torch.Tensor, the accumulated log determinant of the Jacobian
        """
        x2, x1 = self.split(x)              # the order is reversed
        log_scale, shift = self.net(x1)
        scale = torch.exp(log_scale)

        z1, z2 = x1, x2 * scale + shift
        z = torch.cat([z1, z2], dim=-1)

        log_det_J += torch.sum(log_scale, dim=-1)

        return z, log_det_J
    
    def reverse(self, z, log_det_J):
        z1, z2 = self.split(z)
        log_scale, shift = self.net(z1)
        scale = torch.exp(-log_scale)       # use exp(-log_scale) instead of exp(log_scale)

        x1, x2 = z1, (z2 - shift) * scale   # use * instead of /
        x = torch.cat([x2, x1], dim=-1)     # again, reverse the order back

        log_det_J -= torch.sum(log_scale, dim=-1)

        return x, log_det_J


#===================================================================
# modules for NICE

class AdditiveCouplingLayer(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        mode: bool           # indicate which part will be unchanged
    ):
        super().__init__()
        self.net = net
        self.mode = mode

    def forward(self, x):
        if self.mode:
            on, off = x.chunk(2, dim=1)
        else:
            off, on = x.chunk(2, dim=1)
        
        shift = self.net(off)
        on = on + shift

        if self.mode:
            z = torch.cat([on, off], dim=1)
        else:
            z = torch.cat([off, on], dim=1)
        
        return z
    
    def reverse(self, z):
        if self.mode:
            on, off = z.chunk(2, dim=1)
        else:
            off, on = z.chunk(2, dim=1)
        
        shift = self.net(off)
        on = on - shift

        if self.mode:
            x = torch.cat([on, off], dim=1)
        else:
            x = torch.cat([off, on], dim=1)
        
        return x


class ScalingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, dim), requires_grad=True)

    def forward(self, x):
        # because the other Jacobian is 0, 
        # the log determinant of the Jacobian is just the last scale
        log_det_J = torch.sum(self.scale)
        x = x * torch.exp(self.scale)
        return x, log_det_J

    def reverse(self, x):
        x = x * torch.exp(-self.scale)
        return x


#===================================================================



if __name__ == '__main__':
    # test AdditiveCouplingLayer

    net = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 2)
    )
    layer = AdditiveCouplingLayer(net, mode=False)
    x = torch.randn(1, 4)
    z = layer(x)
    x_recon = layer.reverse(z)
    print(torch.allclose(x, x_recon))
    


