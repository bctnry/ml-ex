import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model import VAE

def vae_loss(recon, x, mu, logvar, kl_weight=1e-4):
    recon_l = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))
    return recon_l + kl_weight * kl, recon_l, kl

import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def to_pm1(x):
    return 2 * x - 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(to_pm1)
])
ds = datasets.CIFAR10('data', train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True,
                    generator=torch.Generator().manual_seed(114514))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import lpips
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def vae_loss_perceptual(recon, x, mu, logvar, kl_weight=1e-4, perc_weight=0.5):
    recon_l = F.mse_loss(recon, x)
    perc = lpips_fn(recon, x).mean()
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))
    return recon_l + perc_weight * perc + kl_weight * kl, recon_l, perc, kl

torch.manual_seed(114514)          # weight init
model = VAE(z_ch=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

import pathlib
resume = pathlib.Path('vae_cifar.pt')
if resume.exists():                # continue, don't restart-and-overwrite (B5 lesson)
    chkp = torch.load(resume, map_location=device)
    model.load_state_dict(chkp['model'])
    print('resumed from vae_cifar.pt')

for epoch in range(15):
    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, rl, perc, kl = vae_loss_perceptual(recon, images, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'vae epoch {epoch+1}: recon={rl.item():.4f} kl={kl.item():.2f}')
torch.save({'model': model.state_dict()}, 'vae_cifar.pt')
