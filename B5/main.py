import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import CondUNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.multiprocessing.set_start_method('fork')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

T = 1000
betas = linear_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

NULL_LABEL = 10

sqrt_alphas_cumprod = alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

# adding noise (noised @time t)
def forward_diffusion(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_ac = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_omac = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    x_t = sqrt_ac * x_0 + sqrt_omac * noise
    return x_t, noise

def train_step(model, x_0, y, optimizer):
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=device)
    x_t, noise = forward_diffusion(x_0, t)

    # label dropout: 10% -> null label
    y_in = y.clone().to(device)
    drop = torch.rand(B, device=device) < 0.1
    y_in[drop] = NULL_LABEL

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        noise_pred = model(x_t, t, y_in)
        loss = F.mse_loss(noise_pred.float(), noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)
])
train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True)

model = CondUNet(in_ch=1, d_time=256, n_classes=11).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=2e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

import pathlib
p = pathlib.Path('ddpm_mnist.pt')
if p.exists():
    chkp = torch.load(p, map_location=device)
    model.load_state_dict(chkp['model'])

# n_epochs = 20
n_epochs = 1
for epoch in range(n_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        loss = train_step(model, images, labels, optimizer)
        total_loss += loss
        print('.', end='', flush=True)
    print('')
    avg_loss = total_loss / len(train_loader)
    print(f'epoch {epoch+1}: loss={avg_loss:.4f}')


torch.save({'model': model.state_dict()}, 'ddpm_mnist.pt')


