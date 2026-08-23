import torch
from model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

latent_dim = 16

model = VAE(latent_dim)
chkp = torch.load('vae_mnist_model.pt')
model.load_state_dict(chkp['model'])
model = model.to(device)

@torch.no_grad()
def sample(model, n=16):
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    samples = model.decoder(z)
    return samples


def sample_to_images(images):
    a = (images * 255).to(torch.int)
    ilen = len(a)
    for i, k in enumerate(a):
        with open(f'out_{i:04}.pgm', 'w') as f:
            f.write('P2\n28 28\n255\n')
            for row in range(28):
                for col in range(28):
                    f.write(f"{k[0][row][col]} ")
                f.write('\n')

images = sample(model, 16)
sample_to_images(images)

