"""
Code Reference:
- https://github.com/ermongroup/ncsnv2/blob/master/runners/ncsn_runner.py
- https://github.com/ermongroup/ncsnv2/blob/master/models/__init__.py
"""

import torch
import torch.nn as nn
import numpy as np
from utils import *
from . import Abstract_Model

class NCSNv2(Abstract_Model):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False,
        noise_levels=788,
        sigma_begin=140,
        sigma_end=0.01,
        sigma_dist='geometric',
        ema=True,
        ema_rate=0.999,
        step_lr=0.0000049,
        n_steps_each=4,
    ):
        super(NCSNv2, self).__init__(backbone, backend, backbone_created, backend_created)

        self.sigmas = self.get_sigmas(noise_levels, sigma_begin, sigma_end, sigma_dist)
        self.ema = ema
        if self.ema:
            self.ema_helper = EMAHelper(mu=ema_rate)

        self.step_lr = step_lr
        self.n_steps_each = n_steps_each

    def everything_to(self, device):
        self.sigmas = self.sigmas.to(device)

        # this is a compromise, the ema_helper need to register the backbone to the device
        # if register the backbone at init, it was still on cpu
        if self.ema:
            self.ema_helper.register(self.backbone)

    def get_sigmas(self, noise_levels, sigma_begin, sigma_end, sigma_dist):
        if sigma_dist == 'geometric':
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), noise_levels))
            ).float()
        elif sigma_dist == 'uniform':
            sigmas = torch.tensor(
                np.linspace(sigma_begin, sigma_end, noise_levels)
            ).float()
        else:
            raise NotImplementedError("sigma distribution not supported!")
        return sigmas

    def compute_loss(self, scores, target, used_sigmas, criterion):
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        return criterion(scores, target, used_sigmas)

    def process_batch(self, batch, criterion, device):
        X = batch[0].to(device)
        inds = torch.randint(0, len(X), (X.shape[0],)).to(device)

        used_sigmas = self.sigmas[inds].view(X.shape[0], *([1] * len(X.shape[1:]))) # shape: (batch_size, 1, 1, 1)

        noise = torch.randn_like(X) * used_sigmas
        perturbed_X = X + noise

        target = - 1 / (used_sigmas ** 2) * noise          # - frac{\tilde x - x}{\sigma^2}
        scores = self.forward(perturbed_X, used_sigmas)    # s(\tilde x, \sigma) = s(\tilde x) / \sigma
        loss = self.compute_loss(scores, target, used_sigmas, criterion)
        return loss

    def forward(self, x, used_sigmas):
        output = self.backbone(x)

        # NCSNv2: incorportating the noise info
        # Parameterize the NCSN with s_\theta(x, \sigma) = s_\theta(x) / \sigma
        # s_\theta(x) is unconditional score
        output = output / used_sigmas                   # s(\tilde x, \sigma) = s(\tilde x) / \sigma

        return output
    
    def post_update(self):
        self.ema_helper.update(self.backbone)

    @torch.no_grad()
    def sample_images(self, img_size, device):
        if self.ema_helper is not None:
            test_backbone = self.ema_helper.ema_copy(self.backbone)
        else:
            test_backbone = self.backbone

        test_backbone.eval()

        init_samples = torch.rand(36, self.config.channels, img_size, img_size, device=device)
        
        all_samples = self.anneal_Langevin_dynamics(init_samples, test_backbone, device)

        del test_backbone
        return all_samples


    @torch.no_grad()
    def anneal_Langevin_dynamics(self, x_mod, test_backbone, device):
        with torch.no_grad():
            for c, sigma in enumerate(self.sigmas.cpu().numpy()):
                inds = torch.ones(x_mod.shape[0], device=device) * c
                inds = inds.long()

                step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2
                for s in range(self.n_steps_each):
                    grad = test_backbone(x_mod)
                    used_sigmas = self.sigmas[inds].view(x_mod.shape[0], *([1] * len(x_mod.shape[1:])))

                    grad = grad / used_sigmas

                    noise = torch.randn_like(x_mod)
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

                    x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                    image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

                    snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                    grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                    # print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        # c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            
            # final step
            last_noise_idx = (len(self.sigmas) - 1) * torch.ones(x_mod.shape[0], device=device)
            last_noise_idx = last_noise_idx.long()

            used_sigmas = self.sigmas[last_noise_idx].view(x_mod.shape[0], *([1] * len(x_mod.shape[1:])))
            x_mod = x_mod + self.sigmas[-1] ** 2 * (test_backbone(x_mod) / used_sigmas)

        return x_mod
    


# EMA Helper Class
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict