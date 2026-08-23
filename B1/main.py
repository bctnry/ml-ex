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
from model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl_loss

latent_dim = 16
batch_size = 128
n_epochs = 20
lr = 1e-3

model = VAE(latent_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss = vae_loss(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}')

torch.save({
    'model': model.state_dict(),
}, 'vae_mnist_model.pt')


