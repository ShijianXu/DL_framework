import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels//2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels*2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    """
    Coupling layer module for normalizing flow.
    """
    def __init__(self, net, mask, in_channels) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            mask (torch.Tensor): Binary mask to apply to the input.
            net (nn.Module): Coupling network, used to compute the scale and shift.
                                    Ouput shape: (batch_size, 2 * in_channels, height, width)
        """
        super().__init__()
        self.net = net
        self.scaling_factor = nn.parameter(torch.zeros(in_channels))

        # Register mask as buffer, it is a tnesor but not a parameter
        # but should be part of the state dict
        self.register_buffer('mask', mask)

        # if use self.mask = mask, the `mask` tensor will not be save or loaded when
        # the model is saved or loaded using `torch.save` and `torch.load`
        # also, if the model is moved to the GPU using the `to(device)` method,
        # the `mask` tensor will not be moved to the GPU along with the other parameters

    def create_mask(self, mask_type):
        mask = torch.zeros(self.in_channels, dtype=torch.float32)
        if mask_type == 'checkerboard':
            mask[::2] = 1
        elif mask_type == 'channel_wise':
            mask[::self.in_channels // 2] = 1
        else:
            raise ValueError('Invalid mask type')
        return mask

    def forward(self, x, log_det_J):
        x1 = x * self.mask
        x2 = x * (1 - self.mask)

        # apply network to masked input
        nn_out = self.net(x1)
        log_scale, shift = nn_out.chunk(2, dim=1)   # split along channel dimension into the scale and shift parts
        scale = torch.exp(log_scale)                # scale must be positive, so we exponentiate

        y2 = x2*scale + shift
        y = y2 * self.mask + x1 * (1 - self.mask)   # ????

        log_det_J += torch.sum(log_scale, dim=(1, 2, 3))

        return y, log_det_J

    def reverse(self, y):
        pass


if __name__ == '__main__':
    pass