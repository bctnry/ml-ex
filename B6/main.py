import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def moons(n, seed):
    g = torch.Generator().manual_seed(seed)
    n1 = n // 2
    theta1 = torch.linspace(0, torch.pi, n1) + torch.randn(n1, generator=g) * 0.08
    m1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
    theta2 = torch.linspace(0, torch.pi, n - n1) + torch.randn(n - n1, generator=g) * 0.08
    m2 = torch.stack([1 - torch.cos(theta2), 0.5 - torch.sin(theta2)], dim=1)
    return torch.cat([m1, m2]).to(device)

data = moons(8192, seed=0)
plt.figure(figsize=(4,4))
plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), s=2)
plt.title('x_1: the data distribution')
plt.savefig('moons.png', dpi=100)

from model import VelocityMLP

model = VelocityMLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch = 256
n_steps = 10000

for step in range(n_steps):
    t = torch.rand(batch, device=device)
    x0 = torch.randn(batch, 2, device=device)
    idx = torch.randint(0, data.shape[0], (batch,), device=device)
    x1 = data[idx]

    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0

    loss = torch.nn.functional.mse_loss(model(xt, t), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f'step {step}: loss={loss.item():.4f}')

torch.save({'model': model.state_dict()}, 'fm_moons.pt')


