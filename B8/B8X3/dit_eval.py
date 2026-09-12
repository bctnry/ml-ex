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

torch.manual_seed(114514)

model = LatentDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                  z_ch=16, latent=16, patch=2)
model_chkp = torch.load('dit_latent_best.pt')
model.load_state_dict(model_chkp['model'])
model = model.to(device)

GLOBAL_SEED = 114514

vae = VAE(z_ch=16).to(device)
vae.load_state_dict(torch.load('vae_celeba.pt', map_location=device)['model'])
vae.eval()

cache = torch.load('celeba_latents.pt')
Z = cache['z'].to(device)
txt_emb = cache['txt_emb'].to(device)
LATENT_SCALE = cache.get('latent_scale', 1.0)

@torch.no_grad()
def sample(model, vae, pos_emb, n_samples, neg_emb=None, w=2.0, n_steps=50, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 16, 16, 16, device=device)   # 16x16 latents (64² f4)
    pos = pos_emb.view(1, 1, -1).to(device)
    neg = model.null_txt.expand(1, 1, -1) if neg_emb is None else neg_emb.view(1, 1, -1)
    neg = neg.to(device)
    for i in range(n_steps):
        t = torch.full((2 * n_samples,), i/n_steps, device=device)
        v_c, v_u = model(torch.cat([z, z]),
                         t,
                         torch.cat([pos, neg], dim=0)).chunk(2)
        v = v_u + w * (v_c - v_u)
        z = z + (1/n_steps) * v
    return (vae.dec(z * LATENT_SCALE) + 1) / 2        # (B, 3, 64, 64) in [0,1]

# ---------------------------------------------------------------------------
# Latent-distribution audit: does the generator emit z's with the same per-channel
# statistics as the encoder's cached latents? A mismatch here = off-manifold decode
# (the decoder was only ever trained on encoder-produced latents), and the standard
# fix is per-channel rescaling at decode time (SD's 0.18215 trick, generalized).
@torch.no_grad()
def latent_audit(n_gen=256, w=1.0, n_steps=50, seed=0):
    print('=== latent-distribution audit ===')
    # reference: cached TRAIN latents (already divided by LATENT_SCALE at cache time)
    ref_mean = Z.mean(dim=(0, 2, 3))          # (16,)
    ref_std  = Z.std(dim=(0, 2, 3))           # (16,)
    print('cached  Z: per-ch mean range [%.2f, %.2f]  std range [%.2f, %.2f]' % (
        ref_mean.min(), ref_mean.max(), ref_std.min(), ref_std.max()))

    # generated: run the sampler in latent space (before decode)
    idx = torch.randperm(txt_emb.shape[0], generator=torch.Generator().manual_seed(seed))[:n_gen]
    pos_embs = txt_emb[idx.to(device)]
    torch.manual_seed(seed)
    z = torch.randn(n_gen, 16, 16, 16, device=device)
    pos = pos_embs.unsqueeze(1)
    nul = model.null_txt.expand(n_gen, 1, -1)
    for i in range(n_steps):
        t = torch.full((2 * n_gen,), i / n_steps, device=device)
        v_c, v_u = model(torch.cat([z, z]), t, torch.cat([pos, nul], dim=0)).chunk(2)
        v = v_u + w * (v_c - v_u)
        z = z + (1 / n_steps) * v

    gen_mean = z.mean(dim=(0, 2, 3))          # (16,)
    gen_std  = z.std(dim=(0, 2, 3))           # (16,)
    print('generated: per-ch mean range [%.2f, %.2f]  std range [%.2f, %.2f]' % (
        gen_mean.min(), gen_mean.max(), gen_std.min(), gen_std.max()))
    ratio = (gen_std / ref_std.clamp_min(1e-6))
    print('per-ch std ratio (gen/ref):')
    print('  ', ' '.join(f'{r:.2f}' for r in ratio.tolist()))
    print(f'  mean ratio {ratio.mean():.3f}  min {ratio.min():.3f}  max {ratio.max():.3f}')
    print('  (1.0 = perfect; >>1 = generator z too loud; <<1 = too quiet per channel)')
    return ratio

import datasets
# FID real reference = HELD-OUT split (valid: 19,868 faces, zero overlap with training).
# Fake conditions still come from the TRAIN cache — real=held-out, fake=train-distribution
# is the correct asymmetry (measures generalization).
test_ds = datasets.load_dataset('flwrlabs/celeba', split='valid')
# files are grouped by identity -> first-N is NOT a random face sample.
# Fixed-seed random subset of the whole valid split for the FID reference:
import numpy as np
_fid_sel = np.random.RandomState(GLOBAL_SEED).choice(len(test_ds), size=2000, replace=False)
test_ds = test_ds.select(_fid_sel)

from torchvision import transforms
from torch.utils.data import DataLoader

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
test_loader = DataLoader(HFImageDataset(test_ds, transform),
                         batch_size=128,
                         shuffle=False,      # FID real side: fixed order, fixed set
                         num_workers=4,
                         pin_memory=True)

@torch.no_grad()
def sample_batch(pos_embs, w=2.0, n_steps=25, seed=0):
    # pos_embs :: (B, 384) — B different prompts, one per sample
    torch.manual_seed(seed)
    B = pos_embs.shape[0]
    z = torch.randn(B, 16, 16, 16, device=device)     # 16x16 latents (64² input, f4)
    pos = pos_embs.unsqueeze(1)                       # (B, 1, 384)
    nul = model.null_txt.expand(B, 1, -1)             # (B, 1, 384)
    for i in range(n_steps):
        t = torch.full((2 * B,), i / n_steps, device=z.device)
        v_c, v_u = model(torch.cat([z, z]),
                         t,
                         torch.cat([pos, nul], dim=0)).chunk(2)
        v = v_u + w * (v_c - v_u)
        z = z + (1 / n_steps) * v
    z_out = z
    if ZCH_STD is not None:                       # per-channel match to cached stats
        z_out = (z_out - z_out.mean(dim=(0, 2, 3), keepdim=True)) \
                / z_out.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6) \
                * ZCH_STD[None, :, None, None] + ZCH_MEAN[None, :, None, None]
    return (vae.dec(z_out * LATENT_SCALE) + 1) / 2        # (B, 3, 64, 64) in [0, 1]

@torch.no_grad()
def fid_score(n_fake=5000, batch=64, w=2.0, n_steps=25):
    # batch=64 (not 128): peak VRAM = DiT+VAE + gen batch x2 + VAE decode + Inception.
    # On a 4 GB card, batch 128 across all phases OOMs; 64 peaks ~2.4 GB.
    fid_metric.reset()                                # metric ACCUMULATES — reset per eval!

    # real side: stream the test split (fixed count, fixed order), small Inception batches
    n = 0
    for batch_imgs in test_loader:          # dataset yields x only (no labels)
        for s in range(0, batch_imgs.shape[0], batch):
            fid_metric.update(batch_imgs[s:s+batch].to(device), real=True)
        n += batch_imgs.shape[0]
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
        fid_metric.update(fake, real=False)           # fake is (B,3,64,64) on device
        del fake
        print('.', end='', flush=True)
    print()

    return fid_metric.compute().item()

ZCH_MEAN, ZCH_STD = None, None
# --- run the audit first: is the generator's latent distribution off? ---
_audit_ratio = latent_audit(n_gen=256, w=1.0, n_steps=50, seed=GLOBAL_SEED)
if (_audit_ratio.mean() - 1).abs() > 0.15 or (_audit_ratio.max() - _audit_ratio.min()) > 1.0:
    print('>>> latent statistics mismatch detected — applying per-channel rescale')
    ZCH_MEAN = Z.mean(dim=(0, 2, 3)).to(device)
    ZCH_STD = Z.std(dim=(0, 2, 3)).to(device)
else:
    print('>>> latent statistics roughly match — no rescale needed')
    ZCH_MEAN, ZCH_STD = None, None

# --- FID sweep ---
steps = [25, 50]
print('evaluating...')
n_fake = 2000
for step in steps:
    for w in [0.0, 1.0, 2.0]:
        fid_result = fid_score(n_fake=n_fake, w=w, n_steps=step)
        print(f'FID@w={w},step={step}: {fid_result:.2f}')

    
from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')   # 22M params, frozen, CPU-fast
# embeds :: (L_sentences, 384) — L2-normalized, unit scale, ready to use
def embed_prompts(prompts):
    # prompts :: list[str]
    return torch.tensor(text_encoder.encode(prompts))   # (L, 384)
positive_prompt = "young, blonde hair"
positive_pos_emb = embed_prompts(positive_prompt)
negative_prompt = ''
negative_pos_emb = embed_prompts(negative_prompt) if negative_prompt.strip() else None

import matplotlib.pyplot as plt
ws = [0.0, 1.0, 2.0]
fig, axes = plt.subplots(len(steps), len(ws), figsize=(15, 7))
for r, step in enumerate(steps):
    for c, w in enumerate(ws):
        im = sample(model, vae, positive_pos_emb, 1, neg_emb=negative_pos_emb, w=w, seed=114514, n_steps=step)[0]
        ax = axes[r, c]
        ax.imshow(im.permute(1, 2, 0).cpu().clamp(0, 1))
        ax.set_title(f'w={w}', fontsize=11, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
fig.suptitle(f'result of prompt: +{repr(positive_prompt)} -{repr(negative_prompt)}', fontsize=13)
plt.subplots_adjust(wspace=0.05, hspace=0.08, top=0.86, bottom=0.02, left=0.02, right=0.98)
plt.savefig('prompt_samples.png', dpi=110)
