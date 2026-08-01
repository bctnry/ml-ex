import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, n_heads * self.d_head)
        self.Wk = nn.Linear(d_model, n_heads * self.d_head)
        self.Wv = nn.Linear(d_model, n_heads * self.d_head)
        self.Wo = nn.Linear(n_heads * self.d_head, d_model)

    def forward(self, x, infer=False):
        batch, n_token, d_model = x.shape
        n_heads, d_head = self.n_heads, self.d_head

        Q = self.Wq(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        K = self.Wk(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        V = self.Wv(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (d_head ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        out = out.transpose(1, 2).reshape(batch, n_token, n_heads * d_head)
        return self.Wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, infer=False):
        attn = self.attn(self.ln1(x), infer=infer)
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.d_model = d_model
        # patch embedding: 4x4 pixels x3 channels to vector of shape (d_model,)
        self.patch_proj = nn.Linear(48, d_model)
        # 65 because we appended a [CLS] token at the front.
        self.W_pos = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, 10)

    def forward(self, x, infer=False):
        # x :: (batch, 3, 32, 32)
        batch = x.shape[0]
        # patches :: (batch, 48, 64)  (64 patches of 4x4x3 images)
        patches = F.unfold(x, kernel_size=4, stride=4)
        # patches :: (batch, 64, 48)
        patches = patches.transpose(1, 2)
        # x :: (batch, 64, d_model)
        x = self.patch_proj(patches)
        # cls :: (1, 1, d_model) ===> (batch, 1, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        # x :: (batch, 65, d_model)
        x = torch.cat([cls, x], dim=1)
        x = x + self.W_pos
        for i, block in enumerate(self.blocks):
            x = block(x, infer=infer)
        x = self.ln_f(x)
        x = x[:, 0, :]
        x = self.classify(x)
        return x
    

