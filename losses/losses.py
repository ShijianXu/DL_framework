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

#===================================================================

class NICELoss(nn.Module):
    """
    Loss for NICE model
    """
    def __init__(self):
        super().__init__()

    def forward(self, prior, z, log_det_J):
        """
        determinant is aone real number, it should be of shape (batch_size, )
        For NICE, all batch elements share one scaling layer, so the determinant
        is just one value. But normally it should be of shape (batch_size, ).
        hence log_p_z should be summed over all dimensions except the batch dimension
        """
        log_p_z = prior.log_prob(z).sum(dim=1)

        log_p_x = log_p_z + log_det_J
        nll_loss = -log_p_x
        
        return {
            "loss": nll_loss.mean(),
            "log_likelihood": log_p_x.mean(),
            "log_det": log_det_J.mean()
        }
    

class RealNVPLoss(nn.Module):
    """
    NLL Loss function for RealNVP
    """
    def __init__(self):
        super().__init__()

    def forward(self, prior, z, log_det_J):
        log_p_z = prior.log_prob(z).sum(dim=(1,2,3))
        log_p_x = log_p_z + log_det_J
        nll_loss = -log_p_x
        
        # Calculating bits per dimension
        bpd = nll_loss * torch.log2(torch.exp(torch.tensor(1.0))) / torch.prod(torch.tensor(z.shape[1:]))

        return {
            "nll": nll_loss.mean(),
            "log_likelihood": log_p_x.mean(),
            "log_det": log_det_J.mean(),
            "loss": bpd.mean()
        }


class IrisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prior, z, log_det_J):
        log_p_z = prior.log_prob(z)
        log_p_x = log_p_z + log_det_J
        nll_loss = -log_p_x
        
        return {
            "loss": nll_loss.mean(),
            "log_likelihood": log_p_x.mean(),
            "log_det": log_det_J.mean()
        }


#===================================================================

# TODO: Check correctness of the loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, z_i, z_j):
        pos_sim = F.cosine_similarity(z_i, z_j, dim=-1) / self.tau
        neg_sim = F.cosine_similarity(
            z_i.unsqueeze(1).repeat(1, z_i.shape[0], 1),
            z_j.unsqueeze(0).repeat(z_i.shape[0], 1, 1),
            dim=-1
        ) / self.tau

        neg_sim.fill_diagonal_(float('-inf'))

        loss_e2t = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(neg_sim), dim=-1)).mean()
        loss_t2e = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(neg_sim), dim=0)).mean()
        loss = (loss_e2t + loss_t2e) / 2

        return {
            "loss": loss,
            "loss_e2t": loss_e2t,
            "loss_t2e": loss_t2e
        }
    

class ContrastiveLoss2(nn.Module):
    def __init__(self, tau=0.07, use_softplus=True):
        super().__init__()
        self.tau = tau
        self.use_softplus = use_softplus
        if self.use_softplus:
            self.softplus = nn.Softplus()

    def forward(self, z_i, z_j):
        # compute positive pair similarity
        pos_sim = F.cosine_similarity(z_i, z_j, dim=-1) / self.tau

        # compute negative pair similarity
        neg_sim_i = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=-1) / self.tau
        neg_sim_j = F.cosine_similarity(z_j.unsqueeze(0), z_i.unsqueeze(1), dim=-1) / self.tau

        # exclude the diagonal (postive pairs) from the negative pair similarity
        neg_sim_i = torch.tril(neg_sim_i, diagonal=-1)  # get lower triangular part from the line below the main diagonal
        neg_sim_j = torch.tril(neg_sim_j, diagonal=-1)

        # 计算损失
        exp_pos_sim = torch.exp(pos_sim)
        exp_neg_sim_i = torch.exp(neg_sim_i)
        exp_neg_sim_j = torch.exp(neg_sim_j)

        # 使用softmax或softplus来计算归一化的负样本对概率
        if self.use_softplus:
            neg_prob_i = self.softplus(neg_sim_i).mean(dim=1)
            neg_prob_j = self.softplus(neg_sim_j).mean(dim=1)
        else:
            neg_prob_i = F.softmax(neg_sim_i, dim=1)
            neg_prob_j = F.softmax(neg_sim_j, dim=1)

        loss_e2t = -torch.log(exp_pos_sim / (exp_pos_sim + neg_prob_i.sum(dim=-1))).mean()
        loss_t2e = -torch.log(exp_pos_sim / (exp_pos_sim + neg_prob_j.sum(dim=-1))).mean()
        loss = (loss_e2t + loss_t2e) / 2

        return {
            "loss": loss,
            "loss_e2t": loss_e2t,
            "loss_t2e": loss_t2e
        }
    

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    loss = ContrastiveLoss2()
    z_i = torch.randn(32, 768)
    z_j = torch.randn(32, 768)

    print(loss(z_i, z_j))
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")