import torch
import torch.nn as nn
import torch.nn.functional as F


def anneal_dsm_score_estimation(model, samples, sigmas, labels=None, anneal_power=2.):
    if labels is None:
        # randomly sample noise levels (n_batch, )
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)

        # select corresponding sigmas for noise
        used_sigmas = sigmas[labels].view(samples.shape[0], *([1]*len(samples.shape[1:])))
        noise = torch.randn_like(samples) * used_sigmas
        perturbed_samples = samples + noise

        target = - 1 / (used_sigmas ** 2) * noise

        scores = model(perturbed_samples, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1/2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

        return loss.mean(dim=0)


class Anneal_DSM_Loss(nn.Module):
    def __init__(self, anneal_power=2.):
        super().__init__()
        self.anneal_power = anneal_power

    def forward(self, scores, target, sigma):
        loss = 1/2. * ((scores - target) ** 2).sum(dim=-1) * sigma.squeeze() ** self.anneal_power

        return {
            "loss": loss.mean(dim=0)
        }


class Diffusion_Loss(nn.Module):
    def __init__(self, loss_type="l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, noise, noise_pred):
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, noise_pred)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, noise_pred)
        else:
            raise NotImplementedError()

        return {
            "loss": loss
        }


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