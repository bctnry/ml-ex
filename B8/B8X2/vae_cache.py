import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model import VAE


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
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(z_ch=16).to(device)
p = 'vae_cifar.pt'
chkp = torch.load(p, map_location=device)
state_dict = chkp['model']
model.load_state_dict(state_dict)
model = model.to(device)

# prepare text embeddings
txt_cache = torch.load('txt_cache.pt', map_location=device)
txt_cache = txt_cache['class_emb']

model.eval()
all_z = []
with torch.no_grad():
    for images, _ in loader:
        mu, _ = model.encode(images.to(device))
        all_z.append(mu)
z = torch.cat(all_z)
print('latent std:', z.std().item(), 'mean:', z.mean().item())
LATENT_SCALE = z.std().item()
print(f'LATENT_SCALE: {LATENT_SCALE}')

import numpy as np
model.eval()
latents, labels_all = [], []
embeddings_all = []
i = 0
with torch.no_grad():
    for images, labels in loader:
        g = torch.Generator().manual_seed(114514+i)
        mu, _ = model.encode(images.to(device))
        latents.append((mu / LATENT_SCALE).cpu())
        labels_all.append(labels)
        # pick one variant per image: advanced indexing with aligned index tensors
        # txt_cache :: (10, 6, d_embed);  txt_cache[labels, v] :: (B, d_embed)
        v = torch.randint(0, 6, labels.shape, generator=g)
        embeddings_all.append(txt_cache[labels, v])
        i += 1
Z = torch.cat(latents)
Y = torch.cat(labels_all)
txt_emb = torch.cat(embeddings_all)
torch.save({'z': Z, 'y': Y, 'txt_emb': txt_emb, 'latent_scale': LATENT_SCALE}, 'cifar_latents.pt')
print('cached:', Z.shape)

