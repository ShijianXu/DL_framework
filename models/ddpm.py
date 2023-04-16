# https://github.com/awjuliani/pytorch-diffusion/blob/master/model.py

import torch
import torch.nn as nn


class DenoiseDiffusion():
    def __init__(self, eps_model, n_steps, device) -> None:
        super().__init__()
        self.eps_model = eps_model

        # Create \beta_1, ..., \beta_T linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # \alpha_t = 1 - \beta_t
        self.alpha = 1 - self.beta

        # \alpha_bar_t = \Pi_{s=1:t} \alpha_s
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # total T
        self.n_steps = n_steps

        # \sigma^2 = \beta
        self.simga2 = self.beta

    def q_xt_x0(self, x0, t):
        """
        q(x_t|x_0) = N(x_t; sqrt(alpha_t_bar)*x_0, (1-alpha_t_bar)*I)
        """
        mean = gather()
