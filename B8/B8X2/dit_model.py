import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch=2, in_ch=16, img_size=8, d_model=256):
        # patch: spatial patch size (px of the LATENT image, e.g. 2)
        # in_ch: latent channels (16)
        # -> token dim = patch * patch * in_ch = 2*2*16 = 64
        super().__init__()
        self.patch = patch
        self.n_tokens = (img_size // patch) ** 2        # (8/2)^2 = 16
        self.proj = nn.Linear(patch * patch * in_ch, d_model)
        self.W_pos = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

    def forward(self, x):
        # x :: (B, z_ch, latent, latent) e.g. (B, 16, 8, 8)
        p = F.unfold(x, kernel_size=self.patch, stride=self.patch)
        # p :: (B, patch*patch*z_ch, n_tokens) = (B, 64, 16)
        tokens = p.transpose(1, 2)
        # tokens :: (B, n_tokens, patch*patch*z_ch) = (B, 16, 64)
        return self.proj(tokens) + self.W_pos

class CrossAttention(nn.Module):
    def __init__(self, d_model=256, d_text=384, n_heads=8):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_text, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, txt):
        # x :: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q(self.ln(x)).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv(txt).chunk(2, dim=-1)
        k = k.reshape(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, T, -1)
        return x + self.proj(att)

class DiTBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, mlp_ratio=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model)
        )

        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(4 * d_model, 6 * d_model))
        self.cross = CrossAttention(d_model=d_model, d_text=384)
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, cond, txt):
        # x :: (B, T, d_model), cond :: (B, 4 * d_model)
        B, T, _ = x.shape
        # scale, shift, gate
        s1, b1, g1, s2, b2, g2 = self.mod(cond)[:, None, :].chunk(6, dim=-1)
        # each is (B, 1, d_model)

        # attention branch
        h = self.ln1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head) \
                             .permute(2, 0, 3, 1, 4)
        # (3, B, heads, T, d_head)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, T, -1)
        x = x + g1 * self.proj(att)

        # ffn branch
        h = self.ln2(x) * (1 + s2) + b2
        x = x + g2 * self.ffn(h)
        x = self.cross(x, txt)
        
        return x
    
def timestep_embedding(t, d=256, max_period=10000):
    half = d // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(max_period)) / half))
    args = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class LatentDiT(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=8, n_classes=11,
                 z_ch=16, latent=8, patch=2):
        # n_tokens = (latent // patch) ** 2 = 16
        super().__init__()
        self.d_model = d_model
        self.embed = PatchEmbed(patch=patch, in_ch=z_ch, img_size=latent, d_model=d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, 4 * d_model),
        )
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(4 * d_model, 2 * d_model))
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        self.head = nn.Linear(d_model, patch * patch * z_ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.patch = patch
        self.latent = latent
        self.null_txt = nn.Parameter(torch.randn(1, 1, 384) * 0.02)

    def forward(self, x_t, t, txt):
        # x_t :: (B, z_ch, latent, latent) e.g. (B, 16, 8, 8)
        # t :: (B,)
        # y :: (B,)

        tokens = self.embed(x_t)
        te = self.time_mlp(timestep_embedding(t, d=self.d_model))
        cond = te
        for blk in self.blocks:
            tokens = blk(tokens, cond, txt)
        s, b = self.out_mod(cond)[:, None, :].chunk(2, dim=-1)
        tokens = self.out_norm(tokens) * (1 + s) + b
        v = self.head(tokens)  # (B, n_tokens, patch*patch*z_ch)
        v = v.transpose(1, 2)  # (B, patch*patch*z_ch, n_tokens)
        return F.fold(v, output_size=(self.latent, self.latent),
                      kernel_size=self.patch, stride=self.patch)

            
