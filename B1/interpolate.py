import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

latent_dim = 16

model = VAE(latent_dim)
chkp = torch.load('vae_mnist_model.pt')
model.load_state_dict(chkp['model'])
model = model.to(device)

@torch.no_grad()
def interpolate(model, img1, img2, n_steps=10):
    model.eval()
    mu1, _ = model.encoder(img1)
    mu2, _ = model.encoder(img2)
    res = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = mu1 * (1-alpha) + mu2 * alpha
        samples = model.decoder(z)
        res.append(samples)
    return res


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

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
idx = torch.randint(len(train_ds), (2,))
img1 = (train_ds.data[idx[0]].reshape((1, 1, 28, 28)) / 255).to(device)
img2 = (train_ds.data[idx[1]].reshape((1, 1, 28, 28)) / 255).to(device)
a = interpolate(model, img1, img2, n_steps=16)
sample_to_images(torch.cat((img1, torch.cat(a), img2)))

