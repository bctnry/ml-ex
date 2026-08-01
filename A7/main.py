import os
import pathlib
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from model import ViT

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# def random_crop(images):
#     pad = 4
#     padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
#     B, C, H, W = padded.shape
#     h = torch.randint(0, H - 32, (1,)).item()
#     w = torch.randint(0, H - 32, (1,)).item()
#     return padded[:, :, h:h+32, w:w+32]
# 
# def random_flip(x):
#     flip = (torch.rand(x.shape[0], 1, 1, 1) < 0.5).to(device)
#     return torch.where(flip, x.flip(3), x)

augment_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=10),
])
def image_augment(x):
    # X = random_flip(x)
    # x = random_crop(x)
    return augment_transform(x)

class CIFAR10():
    def __init__(self):
        data_batch_1_p = pathlib.Path(os.getcwd()) / "data_batch_1.pkl"
        data_batch_2_p = pathlib.Path(os.getcwd()) / "data_batch_2.pkl"
        data_batch_3_p = pathlib.Path(os.getcwd()) / "data_batch_3.pkl"
        data_batch_4_p = pathlib.Path(os.getcwd()) / "data_batch_4.pkl"
        data_batch_5_p = pathlib.Path(os.getcwd()) / "data_batch_5.pkl"
        test_batch_p = pathlib.Path(os.getcwd()) / "test_batch.pkl"
        with open(str(data_batch_1_p), 'rb') as f:
            self.data_batch_1 = pickle.load(f)
        with open(str(data_batch_2_p), 'rb') as f:
            self.data_batch_2 = pickle.load(f)
        with open(str(data_batch_3_p), 'rb') as f:
            self.data_batch_3 = pickle.load(f)
        with open(str(data_batch_4_p), 'rb') as f:
            self.data_batch_4 = pickle.load(f)
        with open(str(data_batch_5_p), 'rb') as f:
            self.data_batch_5 = pickle.load(f)
        with open(str(test_batch_p), 'rb') as f:
            self.test_batch = pickle.load(f)
        self.train_dataset = torch.cat([
            torch.tensor(self.data_batch_1["images"]).to(device),
            torch.tensor(self.data_batch_2["images"]).to(device),
            torch.tensor(self.data_batch_3["images"]).to(device),
            torch.tensor(self.data_batch_4["images"]).to(device),
            torch.tensor(self.data_batch_5["images"]).to(device)
        ]).to(torch.bfloat16).to(device)
        self.train_label = torch.cat([
            torch.tensor(self.data_batch_1["labels"]).to(device),
            torch.tensor(self.data_batch_2["labels"]).to(device),
            torch.tensor(self.data_batch_3["labels"]).to(device),
            torch.tensor(self.data_batch_4["labels"]).to(device),
            torch.tensor(self.data_batch_5["labels"]).to(device),
        ]).to(device)
        self.train_dataset_size = self.train_dataset.shape[0]
        self.t10k_dataset = torch.tensor(self.test_batch["images"]).to(torch.bfloat16).to(device)
        self.t10k_label = torch.tensor(self.test_batch["labels"]).to(device)
        self.t10k_dataset_size = self.t10k_dataset.shape[0]

    def get_train_dataset_size(self):
        return self.train_dataset_size

    def get_t10k_dataset_size(self):
        return self.t10k_dataset_size

    def get_train_pair(self, i):
        return (self.train_label[i], self.train_dataset[i])

    def get_t10k_pair(self, i):
        return (self.t10k_label[i], self.t10k_dataset[i])

cifar10 = CIFAR10()
print("CIFAR-10 loaded.")

model = ViT(d_model, n_heads, n_layers)
model = model.to(torch.bfloat16).to(device)

n_epoch = 100
batch_size = 64
steps_per_epoch = cifar10.get_train_dataset_size() // batch_size
n_steps = n_epoch * steps_per_epoch
lr_peak = 5e-4
warmup_steps = 1000
weight_decay = 1e-2

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (n_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_many_train_pair(data, idx):
    return image_augment(data.train_dataset[idx].type(torch.bfloat16)), data.train_label[idx]

def get_many_test_pair(data, idx):
    return data.t10k_dataset[idx].type(torch.bfloat16), data.t10k_label[idx]


def get_batch(data, batch_size):
    idx = torch.randint(0, data.get_train_dataset_size(), (batch_size,))
    return get_many_train_pair(data, idx)

def get_test_batch(data, batch_size, key):
    idx = torch.randint(0, data.get_t10k_dataset_size(), (batch_size,))
    return get_many_test_pair(data, idx)


print('training started')
for epoch in range(n_epoch):
    perm = torch.randperm(cifar10.get_train_dataset_size())
    epoch_loss = 0
    for step in range(steps_per_epoch):
        idx = perm[step * batch_size : (step+1) * batch_size]
        images, labels = get_many_train_pair(cifar10, idx)
        logits = model(images)
        logits = logits.to(torch.float32)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        test_idx = torch.arange(cifar10.get_t10k_dataset_size())
        test_images, test_labels = get_many_test_pair(cifar10, test_idx)
        test_logits = model(test_images)
        test_acc = (test_logits.argmax(-1) == test_labels).float().mean()
        print(f'epoch {epoch+1}: loss={epoch_loss/steps_per_epoch:.4f} test_acc={test_acc:.3f}')
    model.train()

torch.save(model.state_dict(), 'vit_model.pt')
print('model saved.')










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































