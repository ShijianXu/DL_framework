import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import utils


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class SimpleCNF(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width, t0=0, t1=1):
        super().__init__()
        self.func = CNF(in_out_dim, hidden_dim, width)
        self.t0 = t0
        self.t1 = t1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0]).to(self.device),
            covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(self.device),
        )

        # for test/val metric
        self.metric_m = utils.AverageMeter()
        self.best_metric = 0

    def forward(self, x, logp_diff_t1, device, return_last=True):
        z_t, logp_diff_t = odeint(
            self.func,
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
        x = batch.to(device)
        logp_diff_t1 = torch.zeros(x.shape[0], 1).to(device)

        z_t0, logp_diff_t0 = self.forward(x, logp_diff_t1, device)
        
        logp_x = self.p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
        loss = self.compute_loss(logp_x, criterion)

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
    pass