import math
import torch
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = UNet(in_ch=3).to(device)

x = torch.randn(4, 3, 32, 32, device=device)
t = torch.randint(0, 1000, (4,), device=device)

with torch.no_grad():
    out = model(x, t)

assert out.shape == x.shape, "output shape must match input"

t1 = torch.zeros(4, device=device)
t2 = torch.full((4,), 999, device=device)
with torch.no_grad():
    out1 = model(x, t1)
    out2 = model(x, t2)
print(f'{not torch.allclose(out1, out2)}')
