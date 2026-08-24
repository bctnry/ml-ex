import sqlite3
import base64
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as torchmp
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
# from model import VQVAE
from model_ema import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

def vqvae_loss(recon, x, vq_loss):
    recon_loss = F.mse_loss(recon, x)
    return recon_loss + vq_loss

num_codes = 512
dim = 64
batch_size = 128
n_epochs = 50
lr = 3e-4

model = VQVAE(num_codes=num_codes, dim=dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.CIFAR10('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    all_idx = set()
    for images, _ in train_loader:
        images = images.to(device)
        recon, vq_loss, idx = model(images)
        with torch.no_grad():
            all_idx.update(idx.unique().tolist())
        loss = vqvae_loss(recon, images, vq_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    util = len(all_idx) / num_codes
            
    print(f'epoch {epoch+1}: loss={loss.item():.4f} codebook usage={util:.2%}')

torch.save({
    'model': model.state_dict(),
}, 'vqvae_cifar10_model.pt')


