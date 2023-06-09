import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion_Loss(nn.Module):
    pass


def Diffusion_Loss_fn(model, x_0, t, forward_sampling_fn, loss_type="l1"):
    x_noisy, noise = forward_sampling_fn(x_0, t, device=x_0.device)
    noise_pred = model(x_noisy, t)
    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, noise_pred)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, noise_pred)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, noise_pred)
    else:
        raise NotImplementedError()

    return loss


class VAE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recons, input, mu, logvar, kld_weight=1):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        vae_loss = recons_loss + kld_weight*kld_loss
        return {
            "loss": vae_loss,
            "recons_loss": recons_loss,
            "KL-Divergence": -kld_loss
        }


def VAE_Loss_fn(recons, input, mu, logvar, kld_weight=1):
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(-0.5 * torch.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

    vae_loss = recons_loss + kld_weight*kld_loss
    return {
        "loss": vae_loss,
        "recons_loss": recons_loss,
        "KL-Divergence": -kld_loss
    }


class RealNVPLoss(nn.Module):
    """
    NLL Loss function for RealNVP
    """
    def __init__(self):
        super().__init__()

        # Create prior distribution for the final latent space
        # typically assumed to be a standard normal distribution
        self.prior = torch.distributions.Normal(loc=0.0, scale=1.0)

    def forward(self, z, log_det):
        log_z = self.prior.log_prob(z).sum(dim=(1,2,3))
        log_likelihood = log_z + log_det
        nll_loss = -torch.mean(log_likelihood)
        
        return {
            "loss": nll_loss,
            "log_likelihood": log_likelihood,
            "log_det": log_det
        }