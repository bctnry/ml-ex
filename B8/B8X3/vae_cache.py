import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model import VAE

GLOBAL_SEED = 114514

import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from torchvision import transforms
from torch.utils.data import DataLoader

import datasets
ds = datasets.load_dataset('flwrlabs/celeba', split='train')

print('CelebA loaded')

# >>>>>>> text embedding
print('loading text encoder...')
from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')   # 22M params, frozen, CPU-fast
# embeds :: (L_sentences, 384) — L2-normalized, unit scale, ready to use
print('text_encoder loaded...')

def embed_prompts(prompts):
    # prompts :: list[str]
    return torch.tensor(text_encoder.encode(prompts))   # (L, 384)

def tag_to_phrase(s):
    return ' '.join(s.lower().split('_'))

TAGS = ds.column_names[2:]
PHRASE = {i:tag_to_phrase(i) for i in TAGS}

def attrs_to_tags(attrs):
    tags = [PHRASE[t] for t in TAGS if attrs[t] == 1 and PHRASE[t]]
    return ', '.join(tags) or 'a face'
# <<<<<<<< text embedding

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
        labels = {k: self.ds[i][k] for k in TAGS}
        return x, attrs_to_tags(labels)
    
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1),
])
loader = DataLoader(HFImageDataset(ds, transform),
                    batch_size=128,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(GLOBAL_SEED))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = VAE(z_ch=16).to(device)
p = 'vae_celeba.pt'
chkp = torch.load(p, map_location=device)
state_dict = chkp['model']
model.load_state_dict(state_dict)
model = model.to(device)

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
latents = []
embeddings = []
i = 0
current_total = 0
total = len(ds)
with torch.no_grad():
    for images, attributes in loader:
        g = torch.Generator().manual_seed(114514+i)
        mu, _ = model.encode(images.to(device))
        latents.append((mu / LATENT_SCALE).cpu())
        embeddings.append(embed_prompts(attributes))
        i += 1
        current_total += len(attributes)
        print(f'{current_total/total:.4%}', flush=True)
Z = torch.cat(latents)
txt_emb = torch.cat(embeddings)
torch.save({'z': Z, 'txt_emb': txt_emb, 'latent_scale': LATENT_SCALE}, 'celeba_latents.pt')
print('cached:', Z.shape, txt_emb.shape)

