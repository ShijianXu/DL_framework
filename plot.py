import numpy as np

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return np.linspace(start, end, timesteps)


betas = linear_beta_schedule(1000)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod_prev = np.append(1, alphas_cumprod[:-1])
betas_tilde = (1- alphas_cumprod_prev) / (1- alphas_cumprod) * betas

import matplotlib.pyplot as plt
# plot betas and beta_tildes
# plt.plot(betas, label='beta')
# plt.plot(betas_tilde, label='beta_tilde')
plt.plot(betas_tilde/betas, label='beta_tilde/beta')
plt.legend()
plt.show()
