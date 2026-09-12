import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dit_model import LatentDiT
from vae_model import VAE
from dit_hyperparameter import d_model, n_heads, n_layers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def sample(model, vae, pos_emb, n_samples, neg_emb=None, w=2.0, n_steps=50, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 16, 8, 8, device=device)
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
    return (vae.dec(z * LATENT_SCALE) + 1) / 2

from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')   # 22M params, frozen, CPU-fast
# embeds :: (L_sentences, 384) — L2-normalized, unit scale, ready to use
def embed_prompts(prompts):
    # prompts :: list[str]
    return torch.tensor(text_encoder.encode(prompts))   # (L, 384)
positive_prompt = input("(positive) >>> ")
positive_pos_emb = embed_prompts(positive_prompt)
negative_prompt = input("(negative) >>> ")
negative_pos_emb = embed_prompts(negative_prompt) if negative_prompt.strip() else None

import matplotlib.pyplot as plt
ws = [0.0, 0.5, 1.0, 2.0, 4.0, 7.0]
fig, axes = plt.subplots(1, len(ws), figsize=(15, 7))
for c, w in enumerate(ws):
    # SAME seed for every column: the grid must vary only in w,
    # otherwise seed noise (15x bigger than the w effect) dominates the comparison
    im = sample(model, vae, positive_pos_emb, 1, neg_emb=negative_pos_emb, w=w, seed=114514)[0]
    ax = axes[c]
    ax.imshow(im.permute(1, 2, 0).cpu().clamp(0, 1))
    ax.set_title(f'w={w}', fontsize=11, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle(f'result of prompt: +{repr(positive_prompt)} -{repr(negative_prompt)}', fontsize=13)
plt.subplots_adjust(wspace=0.05, hspace=0.08, top=0.86, bottom=0.02, left=0.02, right=0.98)
plt.savefig('prompt_samples.png', dpi=110)
