import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from model import VQVAE
from model_ema import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

num_codes = 512
dim = 64

model = VQVAE(num_codes=num_codes, dim=dim)
chkp = torch.load('vqvae_cifar10_model.pt')
model.load_state_dict(chkp['model'])
model = model.to(device)

@torch.no_grad()
def reconstruct(model, images):
    model.eval()
    recon, _, _ = model(images.permute(0, 3, 1, 2))
    return recon


def sample_to_images(images):
    a = images
    ilen = len(images)
    for i, k in enumerate(a):
        k = (k * 255).clamp(0, 255).to(torch.int)
        print(k.shape)
        with open(f'out_{i:04}.pgm', 'w') as f:
            f.write('P3\n32 32\n255\n')
            for row in range(32):
                for col in range(32):
                    p = k[0][row][col]
                    f.write(f"{p[0]} {p[1]} {p[2]} ")
                f.write('\n')

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.CIFAR10('data', train=True, download=True, transform=transform)
idx = torch.randint(len(train_ds), (2,))
img1 = (torch.tensor([train_ds.data[idx[0]]]) / 255).to(device).to(torch.float)
recon_img1 = reconstruct(model, img1)
sample_to_images([img1, recon_img1.permute(0, 2, 3, 1)])
