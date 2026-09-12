import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dit_model import LatentDiT
from vae_model import VAE
from dit_hyperparameter import d_model, n_heads, n_layers

from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

model = LatentDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
p = pathlib.Path('dit_latent_last.pt')
chkp = torch.load(p, map_location=device)
model.load_state_dict(chkp['model'])
model = model.to(device)

vae = VAE(z_ch=16).to(device)
vae_chkp = torch.load('vae_cifar.pt', map_location=device)
vae.load_state_dict(vae_chkp['model'])
vae = vae.to(device)

latent_pt = torch.load('cifar_latents.pt')
LATENT_SCALE = latent_pt['latent_scale']
txt_emb = latent_pt['txt_emb']          # (50000, 384) cached train embeddings

GLOBAL_SEED = 114514
torch.manual_seed(GLOBAL_SEED)

# REAL reference: CIFAR-10 TEST split images, (3, 32, 32) in [0, 1].
# (NOT the latents — FID runs Inception on images. And NOT the train split:
# use test so the reference never coincides with training data.)
transform = transforms.ToTensor()
test_ds = datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)


@torch.no_grad()
def sample_batch(pos_embs, w=2.0, n_steps=25, seed=0):
    # pos_embs :: (B, 384) — B different prompts, one per sample
    torch.manual_seed(seed)
    B = pos_embs.shape[0]
    z = torch.randn(B, 16, 8, 8, device=device)
    pos = pos_embs.unsqueeze(1)                       # (B, 1, 384)
    nul = model.null_txt.expand(B, 1, -1)             # (B, 1, 384)
    for i in range(n_steps):
        t = torch.full((2 * B,), i / n_steps, device=z.device)
        v_c, v_u = model(torch.cat([z, z]),
                         t,
                         torch.cat([pos, nul], dim=0)).chunk(2)
        v = v_u + w * (v_c - v_u)
        z = z + (1 / n_steps) * v
    return (vae.dec(z * LATENT_SCALE) + 1) / 2        # (B, 3, 32, 32) in [0, 1]


@torch.no_grad()
def fid_score(n_fake=5000, batch=64, w=2.0, n_steps=25):
    # batch=64 (not 128): peak VRAM = DiT+VAE + gen batch x2 + VAE decode + Inception.
    # On a 4 GB card, batch 128 across all phases OOMs; 64 peaks ~2.4 GB.
    fid_metric.reset()                                # metric ACCUMULATES — reset per eval!

    # real side: stream the test split (fixed count, fixed order), small Inception batches
    n = 0
    for imgs, _ in test_loader:
        for s in range(0, imgs.shape[0], batch):
            fid_metric.update(imgs[s:s+batch].to(device), real=True)
        n += imgs.shape[0]
        if n >= n_fake:
            break

    # fake side: conditions = random subset of cached TRAIN embeddings
    # generate -> .cpu() immediately (GPU holds ONE chunk at a time) -> update in chunks
    perm = torch.randperm(txt_emb.shape[0])[:n_fake]
    n_done = 0
    for start in range(0, n_fake, batch):
        batch_idx = perm[start:start + batch]
        fake = sample_batch(txt_emb[batch_idx].to(device),
                            w=w, n_steps=n_steps, seed=GLOBAL_SEED + start)
        fid_metric.update(fake, real=False)           # fake is (B,3,32,32) on device
        del fake
        print('.', end='', flush=True)
    print()

    return fid_metric.compute().item()


if __name__ == '__main__':
    import sys
    n_fake = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    fid_result = fid_score(n_fake=n_fake)
    print(f'\nFID (n_fake={n_fake}): {fid_result:.2f}')

    # append to the experiment log
    with open('fid_log.md', 'a') as f:
        import time
        f.write(f'| {time.strftime("%m-%d %H:%M")} | {pathlib.Path("dit_latent_last.pt").name} '
                f'| epoch {chkp.get("epoch", "?")} | {n_fake} | w=2.0 | FID {fid_result:.2f} |\n')
    print('logged to fid_log.md')
