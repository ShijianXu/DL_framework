# code adapted from torchdiffeq/examples/odenet_mnist.py
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
# 29/05/2024, Wednesday

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import utils

# Convolution part
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
    

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut
    

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
    

# ODE part
class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1               # count the number of function evaluations during the ODE solving
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out
    

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)

        # the rtol, atol controls the error tolerance, hence affect the training speed
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe
    
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ConvResODE(nn.Module):
    def __init__(self, 
            num_classes: int = 10,
            in_channels: int = 1,
            num_filters: int = 64,
            downsampling_method: str = 'conv',
            is_odenet: bool = True
        ):
        super(ConvResODE, self).__init__()

        if downsampling_method == 'conv':
            downsampling_layers = [
                nn.Conv2d(in_channels, num_filters, 3, 1),
                norm(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, 4, 2, 1),
                norm(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, 4, 2, 1),
            ]
        elif downsampling_method == 'res':
            downsampling_layers = [
                nn.Conv2d(in_channels, num_filters, 3, 1),
                ResBlock(num_filters, num_filters, stride=2, downsample=conv1x1(num_filters, num_filters, 2)),
                ResBlock(num_filters, num_filters, stride=2, downsample=conv1x1(num_filters, num_filters, 2)),
            ]

        self.is_odenet = is_odenet
        if is_odenet:
            self.feature_layers = [ODEBlock(ODEfunc(num_filters))]
        else:
            self.feature_layers = [ResBlock(num_filters, num_filters) for _ in range(6)]

        fc_layers = [
            norm(num_filters), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), 
            nn.Linear(num_filters, num_classes)
        ]

        self.model = nn.Sequential(*downsampling_layers, *self.feature_layers, *fc_layers)

        # for test/val metric
        self.metric_m = utils.AverageMeter()
        self.best_metric = 0

        # for ODE counting
        self.f_nfe_meter = utils.AverageMeter()     # forward number of function evaluations
        self.b_nfe_meter = utils.AverageMeter()     # backward number of function evaluations

    def forward(self, x):
        out = self.model(x)
        return out

    def compute_loss(self, output, target, criterion):
        loss = criterion(output, target)
        return {"loss": loss}

    def process_batch(self, batch, criterion, device):
        x, target = batch[0].to(device), batch[1].to(device)
        logits = self(x)
        loss = self.compute_loss(logits, target, criterion)


        # compute forward number of function evaluations
        if self.is_odenet:
            nfe_forward = self.feature_layers[0].nfe
            self.feature_layers[0].nfe = 0
            self.f_nfe_meter.update(nfe_forward)

        return loss
    
    def compute_metric(self, source, preds, target, eval_metric):
        if eval_metric is not None:
            metric_value = eval_metric(preds, target)
            self.metric_m.update(metric_value)
        else:
            # compute accuracy
            preds = torch.argmax(preds, dim=1)
            acc = (preds == target).float().mean()
            self.metric_m.update(acc.item())

    def get_metric_value(self):
        return self.metric_m.avg
    
    def is_best_metric(self):
        if self.metric_m.avg > self.best_metric:
            self.best_metric = self.metric_m.avg
            return True
        else:
            return False

    def display_metric_value(self):
        print(f'Acc value: {self.get_metric_value()} | NFE-F: {self.f_nfe_meter.avg}')

    def reset_metric(self):
        self.metric_m.reset()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def everything_to(self, device):
        pass


if __name__ == '__main__':
    model = ConvResODE(downsampling_method='res', num_classes=10, in_channels=1, num_filters=64)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.size())