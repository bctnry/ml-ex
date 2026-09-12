import pathlib
import torch
import torch.nn.functional as F
from vae_model import VAE

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# held-out split for the recon grid (must NOT be training images)
ds = datasets.load_dataset('flwrlabs/celeba', split='valid')

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1),
])

class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.tf = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        return self.tf(self.ds[i]['image'])

# CelebA files are GROUPED BY IDENTITY — the first N valid images are the same few
# people! Sample a random subset (fixed seed) so the grid/FID reference spans identities.
import numpy as np
IDX_SEED = 114514
sel = np.random.RandomState(IDX_SEED).choice(len(ds), size=2048, replace=False)
subset = ds.select(sel)      # HF datasets: materialize the fixed random subset
loader = DataLoader(HFImageDataset(subset, transform),
                    batch_size=64,
                    shuffle=False,    # order fixed by the seeded selection
                    num_workers=4,
                    pin_memory=True)

model = VAE(z_ch=16).to(device)
p = pathlib.Path('vae_celeba.pt')
assert p.exists(), 'train the VAE first (python3 vae_train.py)'
chkp = torch.load(p, map_location=device)
model.load_state_dict(chkp['model'])
model = model.to(device)
model.eval()

# latent scale, if the cache exists (else measure on the fly from this batch)
LS = None
pc = pathlib.Path('celeba_latents.pt')
if pc.exists():
    LATENT_SCALE = torch.load(pc, map_location=device).get('latent_scale')
    print(f'LATENT_SCALE from cache: {LATENT_SCALE:.4f}')
else:
    LATENT_SCALE = None

images = next(iter(loader)).to(device)

with torch.no_grad():
    mu, logvar = model.encode(images)
    recon_mu = model.dec(mu)                       # deterministic recon (the ceiling)
    z = model.reparam(mu, logvar)                  # sampled-z recon (what the DiT sees)
    recon_sampled = model.dec(z)

recon_mu = ((recon_mu + 1) / 2).clamp(0, 1)
recon_sampled = ((recon_sampled + 1) / 2).clamp(0, 1)
orig = ((images + 1) / 2).clamp(0, 1)

mse_mu = F.mse_loss(recon_mu * 2 - 1, images).item()
mse_s = F.mse_loss(recon_sampled * 2 - 1, images).item()
print(f'recon MSE (mu):      {mse_mu:.4f}')
print(f'recon MSE (sampled): {mse_s:.4f}')

# latent stats for the scale sanity check
print(f'latent mu:  mean={mu.mean().item():.4f}  std={mu.std().item():.4f}')
if LATENT_SCALE:
    print(f'LATENT_SCALE {LATENT_SCALE:.4f} vs measured std {mu.std().item():.4f}')

import matplotlib.pyplot as plt
n = 10
fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
for i in range(n):
    axes[0, i].imshow(orig[i].permute(1, 2, 0).cpu())
    axes[1, i].imshow(recon_mu[i].permute(1, 2, 0).cpu())
    axes[2, i].imshow(recon_sampled[i].permute(1, 2, 0).cpu())
for ax in axes.flat:
    ax.axis('off')
axes[0, 0].set_ylabel('original', fontsize=10)
axes[1, 0].set_ylabel('recon (mu)', fontsize=10)
axes[2, 0].set_ylabel('recon (sampled)', fontsize=10)
plt.subplots_adjust(wspace=0.03, hspace=0.03, top=0.95, bottom=0.02, left=0.06, right=0.99)
plt.savefig('vae_recon_grid.png', dpi=110)
print('saved vae_recon_grid.png')

# optional: perceptual sharpness metric (the number that predicts the FID ceiling)
try:
    import lpips
    lp = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        l_mu = lp(recon_mu * 2 - 1, orig * 2 - 1).mean().item()
        l_s = lp(recon_sampled * 2 - 1, images).mean().item()
    print(f'LPIPS (mu):      {l_mu:.4f}')
    print(f'LPIPS (sampled): {l_s:.4f}')
    print('(LPIPS ~0 = sharp; >0.4 = blurry. This is your FID ceiling indicator.)')
except Exception as e:
    print(f'lpips unavailable ({type(e).__name__}) — MSE only')

# FID(recon) — the tokenizer ceiling, logged like everything else
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    with torch.no_grad():
        fid_metric.reset()
        n_real = 0
        for batch_imgs in loader:
            bi = batch_imgs.to(device)
            mu_b, logvar_b = model.encode(bi)
            rec_b = model.dec(mu_b)
            for s in range(0, bi.shape[0], 64):
                fid_metric.update(((bi[s:s+64] + 1) / 2).clamp(0, 1), real=True)
                fid_metric.update(((rec_b[s:s+64] + 1) / 2).clamp(0, 1), real=False)
            n_real += bi.shape[0]
            if n_real >= 2000:
                break
        fid_recon = fid_metric.compute().item()
    print(f'FID(recon): {fid_recon:.2f}   <- the LDM quality CEILING (DiT can never beat this)')
    with open('fid_log.md', 'a') as f:
        import time
        f.write(f'| {time.strftime("%m-%d %H:%M")} | VAE recon ceiling | 2000 | — | FID {fid_recon:.2f} |\n')
except Exception as e:
    print(f'FID(recon) skipped: {type(e).__name__}: {e}')