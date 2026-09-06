import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import CondUNet
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

NULL_LABEL = 10

model = CondUNet(in_ch=1, d_time=256).to(device)
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

@torch.no_grad()
def sample_cfg_ddpm(model, n_samples, y_labels, w=3.0, T=1000):
    model.eval()
    x = torch.randn(n_samples, 1, 32, 32, device=device)
    y = torch.tensor(y_labels, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device)
        y_pair = torch.cat([y, torch.full_like(y, NULL_LABEL)])
        t_pair = torch.cat([t_batch, t_batch])
        eps_pair = model(torch.cat([x, x]), t_pair, y_pair)
        eps_cond, eps_uncond = eps_pair.chunk(2)

        noise_pred = eps_uncond + w * (eps_cond - eps_uncond)

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

@torch.no_grad()
def sample_cfg_ddim(model, n_samples, y_labels, w=3.0, n_steps=50):
    model.eval()
    x = torch.randn(n_samples, 1, 32, 32, device=device)
    y = torch.tensor(y_labels, device=device)

    stride = T // n_steps
    steps = list(range(stride-1, T, stride))
    steps = steps[::-1] + [0]

    for i in range(len(steps) - 1):
        t, t_next = steps[i], steps[i+1]
        t_batch = torch.full((n_samples,), t, device=device)
        
        y_pair = torch.cat([y, torch.full_like(y, NULL_LABEL)])
        t_pair = torch.cat([t_batch, t_batch])
        eps_pair = model(torch.cat([x, x]), t_pair, y_pair)
        eps_cond, eps_uncond = eps_pair.chunk(2)
        noise_pred = eps_uncond + w * (eps_cond - eps_uncond)

        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_next = alphas_cumprod[t_next]

        x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * noise_pred

    return x

@torch.no_grad()
def sample_cfg_dpm2(model, n_samples, y_labels, w=3.0, n_steps=20):
    model.eval()
    x = torch.randn(n_samples, 1, 32, 32, device=device)
    y = torch.tensor(y_labels, device=device)

    stride = T // n_steps
    steps = list(range(stride-1, T, stride))
    steps = steps[::-1] + [0]

    lam_table = (0.5 * alphas_cumprod / (1 - alphas_cumprod)).log().cpu().numpy()
    neg_lam, ts = -lam_table, np.arange(T)

    def lambd(ab):
        ab = ab.clamp(1e-8, 1-1e-8)
        return 0.5 * (ab / (1 - ab)).log()

    def cfg_eps(xb, tval):
        t_pair = torch.cat([tval, tval])
        y_pair = torch.cat([y, torch.full_like(y, NULL_LABEL)])
        ec, eu = model(torch.cat([xb, xb]), t_pair, y_pair).chunk(2)
        return eu + w * (ec - eu)

    for i in range(len(steps) - 1):
        t, t_next = steps[i], steps[i+1]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_next = alphas_cumprod[t_next]
        lam_t = lambd(alpha_bar_t)
        lam_next = lambd(alpha_bar_next)
        h = lam_next - lam_t
        eps_t = cfg_eps(x, torch.full((n_samples,), float(t), device=device))

        lam_u = lam_t + h / 2
        alpha_bar_u = torch.sigmoid(2 * lam_u)
        t_u = float(np.interp(-lam_u.item(), neg_lam, ts))
        t_u_b = torch.full((n_samples,), t_u, device=device)
        x_u = (alpha_bar_u / alpha_bar_t).sqrt() * x - (1 - alpha_bar_u).sqrt() * (h/2).expm1() * eps_t
        eps_u = cfg_eps(x_u, t_u_b)

        x = (alpha_bar_next / alpha_bar_t).sqrt() * x - (1 - alpha_bar_next).sqrt() * h.expm1() * eps_u

    return x


classes = list(range(10))
weights = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
fig, axes = plt.subplots(10, len(weights), figsize=(3 * len(weights), 30))
for r, cls in enumerate(classes):
    for c, w in enumerate(weights):
        torch.manual_seed(114514 + r)
        s = sample_cfg_dpm2(model, 1, [cls], w=w)
        img = (s[0, 0].cpu() + 1) / 2
        axes[r, c].imshow(img, cmap='gray')
        axes[r, c].axis('off')
        if r == 0:
            axes[r, c].set_title(f'w={w}')
plt.savefig('cfg_grid.png')

