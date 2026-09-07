import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch=4, in_ch=3, d_model=256):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch * in_ch, d_model)
        # 64 patches for 32x32 images w/ 4x4 patches
        self.W_pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

    def forward(self, x):
        # x :: (B, 3, 32, 32)
        B = x.shape[0]
        p = F.unfold(x, kernel_size=self.patch, stride=self.patch)
        tokens = p.transpose(1, 2)
        return self.proj(tokens) + self.W_pos

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
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, cond):
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
        return x
    
def timestep_embedding(t, d=256, max_period=10000):
    half = d // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(max_period)) / half))
    args = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class DiT(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=8, n_classes=11, patch=4):
        super().__init__()
        self.embed = PatchEmbed(patch=patch, d_model=d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, 4 * d_model),
        )
        self.class_emb = nn.Embedding(n_classes, d_model)
        self.class_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, 4 * d_model),
        )
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(4 * d_model, 2 * d_model))
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        self.head = nn.Linear(d_model, patch * patch * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t, t, y):
        # x_t :: (B, 3, 32, 32)
        # t :: (B,)
        # y :: (B,)

        tokens = self.embed(x_t)
        te = self.time_mlp(timestep_embedding(t, d=256))
        ce = self.class_mlp(self.class_emb(y))
        cond = te + ce
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        s, b = self.out_mod(cond)[:, None, :].chunk(2, dim=-1)
        tokens = self.out_norm(tokens) * (1 + s) + b
        v = self.head(tokens)
        v = v.transpose(1, 2)
        return F.fold(v, output_size=(32,32), kernel_size=4, stride=4)
        
            
        
            
