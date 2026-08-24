import torch
import torch.nn as nn
import torch.nn.functional as F

# encode images as distributions instead of a big vector.
class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, dim, 3, stride=1, padding=1)

    def forward(self, x):
        # x :: (batch_size, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, dim, beta=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.register_buffer('codebook', torch.randn(num_codes, dim))

    def forward(self, z_e):
        # z_e :: (batch_size, dim, H, W)
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        # d :: (B*H*W, num_codes)
        d = (z_e_flat ** 2).sum(-1, keepdim=True) \
            - 2 * z_e_flat @ self.codebook.T \
            + (self.codebook ** 2).sum(-1)
        # idx :: (B*H*W,)
        idx = d.argmin(-1)
        # z_q_flat :: (B*H*W, D)
        z_q_flat = self.codebook[idx]
        z_q_st = z_e_flat + (z_q_flat - z_e_flat).detach()
        z_q = z_q_st.reshape(B, H, W, D).permute(0, 3, 1, 2)

        with torch.no_grad():
            one_hot = F.one_hot(idx, self.num_codes).float()
            code_counts = one_hot.sum(0)
            code_sums = one_hot.T @ z_e_flat
            self.codebook.data.mul_(0.99).add_(0.01 * code_sums / (code_counts.unsqueeze(-1) + 1e-5))
            # dead code revival: restart the code containing 0 with random numbers.
            dead = code_counts < 1
            if dead.any():
                n_dead = dead.sum()
                rand_idx = torch.randint(0, z_e_flat.shape[0], (n_dead,), device=z_e_flat.device)
                self.codebook.data[dead] = z_e_flat[rand_idx]
        
        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())
        loss = self.beta * commitment_loss
        return z_q, loss, idx
        
class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(dim, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def forward(self, z_q):
        x = F.relu(self.deconv1(z_q))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, num_codes=512, dim=64):
        super().__init__()
        self.encoder = Encoder(dim)
        self.vq = VectorQuantizer(num_codes, dim, beta=0.25)
        self.decoder = Decoder(dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx


