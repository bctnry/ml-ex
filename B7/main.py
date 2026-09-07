import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DiT

def to_pm1(x):
    return 2 * x - 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(to_pm1)
])
ds = datasets.CIFAR10('data', train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 256
n_heads = 8
n_layers = 8
lr = 1e-4
weight_decay = 1e-2

model = DiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
print('params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

NULL_LABEL = 10

n_epochs = 30

for epoch in range(n_epochs):
    total, cnt = 0, 0
    for images, labels in loader:
        images = images.to(device)
        y = labels.to(device)
        B = images.shape[0]
        t = torch.rand(B, device=device)
        x0 = torch.randn_like(images)
        x1 = images
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        target = x1 - x0
        y[torch.rand(B, device=device) < 0.1] = NULL_LABEL
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            v = model(xt, t, y)
            loss = torch.nn.functional.mse_loss(v.float(), target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total += loss.item()
        cnt += 1
        print('.', end='', flush=True)
    print('')
    print(f'epoch {epoch+1}: loss={total/cnt:.4f}')
    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'dit_cifar_last.pt')

