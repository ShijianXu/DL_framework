import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torch.nn.functional as F

import utils
import copy
import math

# Define basic layers, concatenating time to input
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
    

class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
    

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def hutchinson_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


class ODEfunc(nn.Module):
    def __init__(self, input_shape, hidden_dims, strides, hutchinson_trace=False):
        super().__init__()
        if len(input_shape) == 1:
            # for toy 2d data, with input shape (2,)
            in_out_dim = input_shape[0]
        
            dim_list = [in_out_dim] + hidden_dims + [in_out_dim]
            layers = []
            activation_fns = []
            for i in range(len(dim_list) - 1):
                layers.append(ConcatLinear(dim_list[i], dim_list[i+1]))
                activation_fns.append(nn.Softplus())
            self.layers = nn.ModuleList(layers)
            self.activation_fns = nn.ModuleList(activation_fns)

        else:
            # for image data, with input shape (C, H, W)
            assert len(strides) == len(hidden_dims) + 1
            hidden_dims = tuple(hidden_dims)
            layers = []
            activation_fns = []
            hidden_shape = input_shape

            for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
                if stride is None:
                    layer_kwargs = {}
                elif stride == 1:
                    layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
                elif stride == 2:
                    layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
                elif stride == -2:
                    layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
                else:
                    raise ValueError('Unsupported stride: {}'.format(stride))
                
                layer = ConcatConv2d(hidden_shape[0], dim_out, **layer_kwargs)
                layers.append(layer)
                activation_fns.append(nn.Softplus())

                hidden_shape = list(copy.copy(hidden_shape))
                hidden_shape[0] = dim_out
                if stride == 2:
                    hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
                elif stride == -2:
                    hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

            self.layers = nn.ModuleList(layers)
            self.activation_fns = nn.ModuleList(activation_fns[:-1])

        self.hutchinson_trace = hutchinson_trace

    def get_z_dot(self, t, z):
        """
        dz_dt = NN(t, z(t))
        """
        dz_dt = z
        for l, layer in enumerate(self.layers):
            dz_dt = layer(t, dz_dt)
            if l < len(self.layers) - 1:
                dz_dt = self.activation_fns[l](dz_dt)
        return dz_dt

    def forward(self, t, x):
        """
        Args:
            t: time
            x: tuple, (z, logp_diff_t)
        """
        z = x[0]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            dz_dt = self.get_z_dot(t, z)

            if self.hutchinson_trace:
                e = sample_rademacher_like(z).to(z)
                tr_df_dz = hutchinson_approx(dz_dt, z, e)
                dlogp_z_dt = -tr_df_dz.view(batchsize, 1)
            else:
                # brute force computation of trace(df/dz), O(D^2)
                dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return dz_dt, dlogp_z_dt


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


class FFJORD(nn.Module):
    def __init__(self, input_shape, hidden_dims, strides=None, t0=0, t1=1, hutchinson_trace=False):
        super().__init__()
        self.input_shape = input_shape
        self.ode_func = ODEfunc(input_shape, hidden_dims, strides, hutchinson_trace)
        self.t0 = t0
        self.t1 = t1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # for sampling
        self.fixed_z = torch.randn(100, *input_shape).to(self.device)

    def forward(self, x, logp_diff_t1, device, return_last=True):
        z_t, logp_diff_t = odeint(
            self.ode_func,
            (x, logp_diff_t1),
            torch.tensor([self.t1, self.t0]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        if return_last:
            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            return z_t0, logp_diff_t0
        else:
            return z_t, logp_diff_t

    def compute_loss(self, logp_x, criterion):
        # loss = -logp_x.mean(0)
        # return {"loss": loss}

        return criterion(logp_x)

    def process_batch(self, batch, criterion, device):
        if len(batch) == 2:
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)

        z_t0, logp_diff_t0 = self.forward(x, logp_diff_t1, device)
        
        logp_z0 = standard_normal_logprob(z_t0).view(z_t0.shape[0], -1).sum(1, keepdim=True)
        logp_x = logp_z0.to(device) - logp_diff_t0.view(-1)
        loss = self.compute_loss(logp_x, criterion)

        return loss
    
    @torch.no_grad()
    def generate(self, num_samples, device):
        """Generate samples by integrating from t0 to t1 starting from the base distribution."""
        # Sample from the base distribution p_z0 at t0

        _logpz = torch.zeros(self.fixed_z.shape[0], 1).to(self.fixed_z)

        # Integrate forward from t0 to t1
        z_t, _ = odeint(
            self.ode_func,
            (self.fixed_z, _logpz),
            torch.tensor([self.t0, self.t1]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        
        # Return the final state, which is a sample from the distribution at t1
        return z_t[-1].view(-1, *self.input_shape)
    
    def compute_metric(self, source, preds, target, eval_metric):
        pass

    def get_metric_value(self):
        return self.metric_m.avg
    
    def is_best_metric(self):
        if self.metric_m.avg > self.best_metric:
            self.best_metric = self.metric_m.avg
            return True
        else:
            return False

    def display_metric_value(self):
        pass

    def reset_metric(self):
        self.metric_m.reset()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def everything_to(self, device):
        pass


if __name__ == "__main__":
    layer = ConcatConv2d(3, 3, padding=1)
    x = torch.randn(1, 3, 30, 30)
    t = torch.tensor([1.2])
    out = layer(t, x)
    print(out.shape)