import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_ch=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 2 * z_ch, 3, 1, 1)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(z_ch, 64, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.z_ch = z_ch

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparam(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar


            
