import torch
import torch.nn as nn
import torch.nn.functional as F

def VAE_Loss(recons, input, mu, logvar, kld_weight=1):
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(-0.5 * torch.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

    vae_loss = recons_loss + kld_weight*kld_loss
    return {
        "loss": vae_loss,
        "recons_loss": recons_loss,
        "KL-Divergence": -kld_loss
    }