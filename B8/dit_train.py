import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_model import LatentDiT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(114514)          # weight init

d_model = 256
n_heads = 8
n_layers = 8
lr = 2e-4
weight_decay = 1e-2

model = LatentDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

NULL_LABEL = 10
cache = torch.load('cifar_latents.pt')
Z, Y = cache['z'].to(device), cache['y'].to(device)

for epoch in range(60):
    perm = torch.randperm(Z.shape[0], device=device)
    total, cnt = 0, 0
    for i in range(0, Z.shape[0], 128):
        idx = perm[i:i+128]
        z1, y = Z[idx], Y[idx]
        B = z1.shape[0]
        t = torch.rand(B, device=device)
        z0 = torch.randn_like(z1)
        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1

        # 10% dropout
        y = y.clone()
        y[torch.rand(B, device=device) < 0.1] = NULL_LABEL

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            v = model(zt, t, y)
            loss = torch.nn.functional.mse_loss(v.float(), z1 - z0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        cnt += 1
        print('.', end='', flush=True)
    print('')
    print(f'epoch {epoch+1}: loss={total/cnt:.4f}')
    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'dit_latent_last.pt')

