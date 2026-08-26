import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, d_time):
        super().__init__()
        self.d_time = d_time
        self.mlp1 = nn.Linear(d_time, d_time * 4)
        self.mlp2 = nn.Linear(d_time * 4, d_time * 4)

    def forward(self, t):
        # t :: (batch_size,)
        half = self.d_time // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (math.log(10000) / half)
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = F.silu(self.mlp1(emb))
        emb = self.mlp2(emb)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_time_emb):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(d_time_emb, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_ch=3, d_time=256):
        super().__init__()
        d_te = d_time * 4
        self.time_embed = TimeEmbedding(d_time)

        self.init_conv = nn.Conv2d(in_ch, 32, 3, padding=1)

        self.down1 = ResBlock(32, 32, d_te)
        self.down2 = ResBlock(32, 32, d_te)
        self.downsample1 = Downsample(32)
        self.down3 = ResBlock(32, 64, d_te)
        self.down4 = ResBlock(64, 64, d_te)
        self.downsample2 = Downsample(64)
        self.down5 = ResBlock(64, 128, d_te)
        self.down6 = ResBlock(128, 128, d_te)
        self.downsample3 = Downsample(128)

        self.mid1 = ResBlock(128, 256, d_te)
        self.mid2 = ResBlock(256, 256, d_te)

        self.upsample3 = Upsample(256)
        self.up6 = ResBlock(256 + 128, 128, d_te)
        self.up5 = ResBlock(128 + 128, 128, d_te)

        self.upsample2 = Upsample(128)
        self.up4 = ResBlock(128 + 64, 64, d_te)
        self.up3 = ResBlock(64 + 64, 64, d_te)
        
        self.upsample1 = Upsample(64)
        self.up2 = ResBlock(64 + 32, 32, d_te)
        self.up1 = ResBlock(32 + 32, 32, d_te)

        self.norm_out = nn.GroupNorm(32, 32)
        self.conv_out = nn.Conv2d(32, in_ch, 1)

    def forward(self, x, t):

        t_emb = self.time_embed(t)

        # x :: (batch_size, 32, 32, 32)
        x = self.init_conv(x)
        
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d = self.downsample1(d2)
        d3 = self.down3(d, t_emb)
        d4 = self.down4(d3, t_emb)
        d = self.downsample2(d4)
        d5 = self.down5(d, t_emb)
        d6 = self.down6(d5, t_emb)
        d = self.downsample3(d6)

        m1 = self.mid1(d, t_emb)
        m2 = self.mid2(m1, t_emb)

        u = self.upsample3(m2)
        u = self.up6(torch.cat([u, d6], dim=1), t_emb)
        u = self.up5(torch.cat([u, d5], dim=1), t_emb)
        u = self.upsample2(u)
        u = self.up4(torch.cat([u, d4], dim=1), t_emb)
        u = self.up3(torch.cat([u, d3], dim=1), t_emb)
        u = self.upsample1(u)
        u = self.up2(torch.cat([u, d2], dim=1), t_emb)
        u = self.up1(torch.cat([u, d1], dim=1), t_emb)

        u = F.silu(self.norm_out(u))
        u = self.conv_out(u)
        return u
        
