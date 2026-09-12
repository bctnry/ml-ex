import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model import VAE

GLOBAL_SEED = 114514

def vae_loss(recon, x, mu, logvar, kl_weight=1e-4):
    recon_l = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))
    return recon_l + kl_weight * kl, recon_l, kl

import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from torchvision import transforms
from torch.utils.data import DataLoader

import datasets
ds = datasets.load_dataset('flwrlabs/celeba', split='train')

class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform, prompt_table=None):
        self.ds = hf_ds
        self.tf = transform
        self.table = prompt_table
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i]['image']
        x = self.tf(img)
        return x
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1),
])
loader = DataLoader(HFImageDataset(ds, transform),
                    batch_size=128,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(GLOBAL_SEED))


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
resume = pathlib.Path('vae_celeba.pt')
if resume.exists():                # continue, don't restart-and-overwrite (B5 lesson)
    chkp = torch.load(resume, map_location=device)
    model.load_state_dict(chkp['model'])
    print('resumed from vae_celeba.pt')

for epoch in range(15):
    for images in loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, rl, perc, kl = vae_loss_perceptual(recon, images, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('.', end='', flush=True)

    print(f'vae epoch {epoch+1}: recon={rl.item():.4f} kl={kl.item():.2f}')
torch.save({'model': model.state_dict()}, 'vae_celeba.pt')
