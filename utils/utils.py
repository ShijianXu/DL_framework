# some utility classes and functions
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def PSNR(input, target):
    """Computes peak signal-to-noise ratio."""
    
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a list of values vals
    while considering the batch dimension of x_shape.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape)-1))).to(t.device)


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, smaple_shapes):
        return torch.distributions.Uniform(0, 1).sample(smaple_shapes).logit().cuda()
    
    def sample2(self, smaple_shapes):
        z = torch.distributions.Uniform(0, 1).sample(smaple_shapes).cuda()
        return torch.log(z) - torch.log(1 - z)
    

if __name__ == "__main__":
    # test StandardLogistic
    dist = StandardLogistic()
    samples = dist.sample2((1,))

    print(samples)