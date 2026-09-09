import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dit_model import LatentDiT
from vae_model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 256
n_heads = 8
n_layers = 8

NULL_LABEL = 10

model = LatentDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
p = pathlib.Path('dit_latent_last.pt')
chkp = torch.load(p, map_location=device)
state_dict = chkp['model']
model.load_state_dict(state_dict)
model = model.to(device)

vae = VAE(z_ch=16).to(device)
vae_p = pathlib.Path('vae_cifar.pt')
vae_chkp = torch.load(vae_p, map_location=device)
vae_state_dict = vae_chkp['model']
vae.load_state_dict(vae_state_dict)
vae = vae.to(device)

latent_pt = torch.load('cifar_latents.pt')
LATENT_SCALE = latent_pt['latent_scale']

@torch.no_grad()
def sample(model, vae, n_samples, y_labels, w=2.0, n_steps=50, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 16, 8, 8, device=device)
    y = torch.tensor(y_labels, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        y_pair = torch.cat([y, torch.full_like(y, NULL_LABEL)])
        t_pair = torch.cat([t, t])
        v_c, v_u = model(torch.cat([z, z]), t_pair, y_pair).chunk(2)
        v = v_u + w * (v_c - v_u)
        z = z + dt * v
    img = vae.dec(z * LATENT_SCALE)
    return (img + 1) / 2

classes = torch.arange(10)
class_names = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
import matplotlib.pyplot as plt
ws = [0.0, 0.5, 1.0, 2.0, 4.0, 7.0]
fig, axes = plt.subplots(len(ws), len(classes), figsize=(15, 7))
for r, w in enumerate(ws):
    for c, cls in enumerate(classes):
        im = sample(model, vae, 1, [cls], w=w, seed=114514 + int(cls))[0]
        ax = axes[r, c]
        ax.imshow(im.permute(1, 2, 0).cpu().clamp(0, 1))
        if r == 0:
            ax.set_title(class_names[int(cls)], fontsize=11)
        if c == 0:
            ax.set_ylabel(f'w={w}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
fig.suptitle('CFG weight (rows) × class (columns)', fontsize=13)
plt.subplots_adjust(wspace=0.05, hspace=0.08, top=0.90, bottom=0.02, left=0.05, right=0.98)
plt.savefig('ldm_samples.png', dpi=110)
