import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import VelocityMLP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VelocityMLP().to(device)
p = pathlib.Path('fm_moons_reflow.pt')
if not p.exists():
    p = pathlib.Path('fm_moons.pt')
chkp = torch.load(p, map_location=device)
state_dict = chkp['model']
model.load_state_dict(state_dict)

@torch.no_grad()
def sample(model, n, n_steps=20, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i / n_steps, device=device)
        x = x + dt * model(x, t)
    return x

samples = sample(model, 4096)
plt.figure(figsize=(4, 4))
plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s = 4, alpha=0.5)
plt.title('v_0 integrates noise -> moons')
plt.savefig('samples.png', dpi=100)

traj_n = 64
torch.manual_seed(1)
x = torch.randn(traj_n, 2, device=device)
path = [x.cpu()]
with torch.no_grad():
    for i in range(20):
        t = torch.full((traj_n,), i / 20, device=device)
        x = x + (1 / 20) * model(x, t)
        path.append(x.cpu())
path = torch.stack(path)

plt.figure(figsize=(5, 5))
for j in range(traj_n):
    plt.plot(path[:, j, 0], path[:, j, 1], '-', c='steelblue', lw=0.8, alpha=0.6)
    plt.scatter(path[0, j, 0], path[0, j, 1], c='red', s=18, zorder=3)    # noise start
    plt.scatter(path[-1, j, 0], path[-1, j, 1], c='green', s=18, zorder=3)  # data end
plt.title('ODE trajectories: noise (red) → data (green)')
plt.savefig('trajectories.png', dpi=100)


import numpy as np
g = np.linspace(-2.5, 2.5, 20)
gx, gy = np.meshgrid(g, g)
pts = torch.tensor(np.stack([gx, gy], -1).reshape(-1, 2), dtype=torch.float32, device=device)
tt = torch.full((pts.shape[0],), 0.8, device=device)   # field at t=0.5
with torch.no_grad():
    v = model(pts, tt).cpu().numpy()
plt.figure(figsize=(5, 5))
plt.quiver(gx.reshape(-1), gy.reshape(-1), v[:, 0], v[:, 1])
plt.title('v(x, t=0.8): the learned velocity field')
plt.savefig('field.png', dpi=100)

@torch.no_grad()
def sample_heun(model, n, n_steps=10, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 2, device=device)
    dt = 1 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i / n_steps, device=device)
        v1 = model(x, t)
        x_mid = x + dt * v1
        t_mid = torch.full((n,), (i + 0.5) / n_steps, device=device)
        v2 = model(x_mid, t_mid)
        x = x + dt * v2
    return x

samples = sample_heun(model, 4096)
plt.figure(figsize=(4, 4))
plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s = 4, alpha=0.5)
plt.title('v_0 integrates noise -> moons')
plt.savefig('samples_heun.png', dpi=100)
