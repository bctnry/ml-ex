import os
import pathlib
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as torchmp
import torchvision.transforms.v2 as T
from model import ViT
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MASTER_ADDR'] = '::1'
os.environ['MASTER_PORT'] = '29500'


d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

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
            torch.tensor(self.data_batch_1["images"]),
            torch.tensor(self.data_batch_2["images"]),
            torch.tensor(self.data_batch_3["images"]),
            torch.tensor(self.data_batch_4["images"]),
            torch.tensor(self.data_batch_5["images"])
        ]).to(torch.bfloat16)
        self.train_label = torch.cat([
            torch.tensor(self.data_batch_1["labels"]),
            torch.tensor(self.data_batch_2["labels"]),
            torch.tensor(self.data_batch_3["labels"]),
            torch.tensor(self.data_batch_4["labels"]),
            torch.tensor(self.data_batch_5["labels"]),
        ])
        self.train_dataset_size = self.train_dataset.shape[0]
        self.t10k_dataset = torch.tensor(self.test_batch["images"]).to(torch.bfloat16)
        self.t10k_label = torch.tensor(self.test_batch["labels"])
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


n_epoch = 10
batch_size = 64
steps_per_epoch = cifar10.get_train_dataset_size() // batch_size
n_steps = n_epoch * steps_per_epoch
lr_peak = 5e-4
warmup_steps = 1000
weight_decay = 1e-2


def train():
    print('setting up...')
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)

    rank = dist.get_rank()
    device_id = rank % torch.accelerator.device_count()
    
    train_ds = TensorDataset(cifar10.train_dataset, cifar10.train_label)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler)
    test_ds = TensorDataset(cifar10.t10k_dataset, cifar10.t10k_label)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    model = ViT(d_model, n_heads, n_layers)
    model = model.to(torch.bfloat16).to(f'cuda:{device_id}')
    ddp_model = DDP(model, device_ids=[device_id])
    optimizer = optim.AdamW(ddp_model.parameters(), lr=lr_peak, weight_decay=weight_decay)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (n_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print('training started')
    for epoch in range(n_epoch):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(f'cuda:{device_id}')
            labels = labels.to(f'cuda:{device_id}')
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = ddp_model(images)
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
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(f'cuda:{device_id}').to(torch.bfloat16)
                labels = labels.to(f'cuda:{device_id}')
                test_logits = model(images)
                correct += (test_logits.argmax(-1) == labels).sum()
                total += labels.size(0)
            test_acc = correct / total
            print(f'epoch {epoch+1}: loss={epoch_loss/steps_per_epoch:.4f} test_acc={test_acc:.3f}')
        model.train()

    dist.barrier()
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), 'vit_model.pt')
        print('model saved.')
    dist.destroy_process_group()

if __name__ == "__main__":
    train()

