import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import VelocityMLP

model = VelocityMLP().to(device)
chkp = torch.load('fm_moons.pt', map_location=device)
state_dict = chkp['model']
model.load_state_dict(state_dict)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# generating data for reflow...
@torch.no_grad()
def sample_pairs(model, n, n_steps=20, seed=0):
    # integrate the ODE from noise to data, KEEPING the (start, end) pairing:
    # reflow's whole point is that x1 = ODE(x0) for the SAME x0.
    torch.manual_seed(seed)
    x0 = torch.randn(n, 2, device=device)
    y = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i / n_steps, device=device)
        y = y + dt * model(y, t)
    return torch.cat([x0, y], dim=1)   # (n, 4): cols 0:2 = x0, cols 2:4 = paired x1

batch = 256
n_steps = 3000

pairs = sample_pairs(model, 8192)

for step in range(n_steps):
    idx = torch.randint(0, pairs.shape[0], (batch,), device=device)
    pair = pairs[idx]
    x0 = pair[:, :2]   # the exact noise the model started from
    x1 = pair[:, 2:]   # the endpoint the model itself produced
    t = torch.rand(batch, device=device)

    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0

    loss = torch.nn.functional.mse_loss(model(xt, t), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f'step {step}: loss={loss.item():.4f}')

torch.save({'model': model.state_dict()}, 'fm_moons_reflow.pt')


