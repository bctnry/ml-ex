import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import UNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

T = 1000
betas = linear_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

model = UNet(in_ch=1, d_time=256).to(device)
chkp = torch.load('ddpm_mnist.pt', map_location=device)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in chkp['model'].items()}
model.load_state_dict(state_dict)

@torch.no_grad()
def sample(model, n_samples, T=1000):
    model.eval()

    x = torch.randn(n_samples, 1, 32, 32, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device)

        noise_pred = model(x, t_batch)

        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        mean = (1 / alpha_t.sqrt()) * (
            x - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * noise_pred
        )

        if t > 0:
            z = torch.randn_like(x)
            x = mean + beta_t.sqrt() * z
        else:
            x = mean
            
    return x

samples = sample(model, 16, T=T)
samples = (samples + 1) / 2
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu(), cmap='gray')
    ax.axis('off')
plt.savefig('ddpm_samples.png')

