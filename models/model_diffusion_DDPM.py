"""
Code Reference:
- https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing
"""

import torch
from utils import *
from . import Abstract_Model

class DiffusionDDPM(Abstract_Model):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False,
        T=200,
        beta_schedule='linear', # 'linear', 'cosine', 'quadratic', 'sigmoid'
    ):
        super(DiffusionDDPM, self).__init__(backbone, backend, backbone_created, backend_created)

        self.T = T

        # define betas
        if beta_schedule == 'linear':
            self.betas = self.linear_beta_schedule(timesteps=T)
        elif beta_schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps=T)
        elif beta_schedule == 'quadratic':
            self.betas = self.quadratic_beta_schedule(timesteps=T)
        elif beta_schedule == 'sigmoid':
            self.betas = self.sigmoid_beta_schedule(timesteps=T)
        else:
            raise NotImplementedError('beta_schedule {} not implemented'.format(beta_schedule))

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)                      # \bar alpha_t = \prod_{i=1}^t (1-\alpha_i)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # \bar alpha_{t-1}
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)                    # sqrt(\bar alpha_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)     # sqrt(1- \bar alpha_t)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def compute_loss(self, noise, noise_pred, criterion):
        return criterion(noise, noise_pred)

    def process_batch(self, batch, criterion, device):
        batch = batch[0].to(device)

        t = torch.randint(0, self.T, (batch.shape[0],), device=device).long()
        x_noisy, noise = self.forward_diffusion_sample(batch, t, device)
        noise_pred = self.forward(x_noisy, t)

        loss = self.compute_loss(noise, noise_pred, criterion)
        return loss

    def forward(self, x, t):
        return self.backbone(x, t)

    def forward_diffusion_sample(self, x_0, t, device):
        """
        Takes an image and a timestep as input and 
        returns the nosiy image for that timestep.
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cimprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # mean + variance
        # x_t= \sqrt \bar alpha_t * x_0 + \sqrt(1-\bar alpha_t) * N(0,1)
        return sqrt_alphas_cumprod_t.to(device) * x_0 + \
            sqrt_one_minus_alphas_cimprod_t.to(device) * noise.to(device), noise.to(device)

    # beta schedules
    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)
    
    def quadratic_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start**0.5, end**0.5, timesteps) ** 2
    
    def sigmoid_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (end - start) + start

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linespace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @torch.no_grad()
    def sample_images(self, img_size, device):
        # an img starts from pure noise
        img = torch.randn(1, 3, img_size, img_size).to(device)

        num_images = 10
        stepsize = int(self.T / num_images)
        image_list = []

        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, dtype=torch.int64, device=device)
            img = self.sample_timestep(img, t)

            img = torch.clamp(img, -1., 1.)
            if i % stepsize == 0:
                image_list.append((img + 1.) / 2.)  # convert to [0, 1] range
        
        image_tensor = torch.cat(image_list, dim=0)
        return image_tensor
    
    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Call the model to predict the noise in the image x_t at time t,
        get the clean image x_t-1.
        Apply extra noise to the denoised image, unless we are at the t=0.
        """
        betas_t = get_index_from_list(self.betas, t, x.shape)            # \beta_t
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(      # sqrt(1- \bar alpha_t)
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(                  # \sqrt(1/alpha_t)
            self.sqrt_recip_alphas, t, x.shape
        )

        # predict noise
        noise_pred = self.forward(x, t)

        # get the mean: x_t - noise_pred
        model_mean = sqrt_recip_alphas_t * \
            (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
