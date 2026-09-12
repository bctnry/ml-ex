import math
import pathlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dit_model import LatentDiT
from vae_model import VAE
from dit_hyperparameter import d_model, n_heads, n_layers

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchmetrics.image.fid import FrechetInceptionDistance
fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

torch.manual_seed(114514)          # weight init

lr_peak = 2e-4
weight_decay = 1e-2
warmup_steps = 500

model = LatentDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)

GLOBAL_SEED = 114514

vae = VAE(z_ch=16).to(device)
vae.load_state_dict(torch.load('vae_cifar.pt', map_location=device)['model'])
vae.eval()
LATENT_SCALE = torch.load('cifar_latents.pt').get('latent_scale', 1.0)

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def cosine_lr(step, warmup=500, total=24000, peak=2e-4, end=1e-5):
    # LambdaLR MULTIPLIES the optimizer base lr (peak) by this lambda -> return RATIOS
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    return (end / peak) + 0.5 * (1 - end / peak) * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

NULL_LABEL = 10
cache = torch.load('cifar_latents.pt')
Z, Y = cache['z'].to(device), cache['y'].to(device)
txt_emb = cache['txt_emb'].to(device)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
def fid_score(n_fake=5000, batch=32, w=2.0, n_steps=25):
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

best = +float('inf')
for epoch in range(60):
    perm = torch.randperm(Z.shape[0], device=device)
    total, cnt = 0, 0
    for i in range(0, Z.shape[0], 128):
        idx = perm[i:i+128]
        z1, y = Z[idx], Y[idx]
        txt = txt_emb[idx].unsqueeze(1)
        B = z1.shape[0]
        t = torch.rand(B, device=device)
        z0 = torch.randn_like(z1)
        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1

        # 10% dropout
        txt = txt.clone()
        drop = torch.rand(B, device=device) < 0.1
        txt[drop] = model.null_txt.expand(drop.sum(), 1, -1) if drop.any() else txt[drop]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            v = model(zt, t, txt)
            loss = torch.nn.functional.mse_loss(v.float(), z1 - z0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total += loss.item()
        cnt += 1
        print('.', end='', flush=True)
    print('')
    print(f'epoch {epoch+1}: loss={total/cnt:.4f}')
    if epoch % 5 == 4:
        print('evaluating...')
        n_fake = 2000
        fid_result = fid_score(n_fake=n_fake)
        if fid_result < best:
            best = fid_result
            torch.save({'model': model.state_dict(), 'epoch': epoch}, 'dit_latent_best.pt')
        print(f'FID@{epoch+1}: {fid_result:.2f} (best {best:.2f})')
        with open('fid_log.md', 'a') as f:
            import time
            f.write(f'| {time.strftime("%m-%d %H:%M")}'
                    f'| epoch {epoch} | {n_fake} | w=2.0 | FID {fid_result:.2f} | best {best:.2f} |\n')
    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'dit_latent_last.pt')

