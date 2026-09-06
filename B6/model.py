import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityMLP(nn.Module):
    def __init__(self, d=2, d_hidden=128):
        super().__init__()
        half = 32
        freqs = torch.exp(-torch.arange(half) * (torch.log(torch.tensor(10000.0)) / half))
        self.register_buffer('freqs', freqs)
        # x (d=2) & time embedding (64)
        self.net = nn.Sequential(
            nn.Linear(d + 64, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d)
        )
        
    def time_emb(self, t):
        # t :: (B,) in [0, 1]
        args = t[:, None] * self.freqs[None, :] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))

    
        

