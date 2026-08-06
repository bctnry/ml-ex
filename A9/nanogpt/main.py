import os
import pathlib
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as torchmp
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NanoGPT
from dataset3 import TokenDataset


d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 128
batch_size = 32
n_steps = 5000
total_steps = 5000
lr_peak = 1e-3
warmup_steps = 500
weight_decay = 1e-2
n_epoch = 1

Tokenizer = tiktoken.get_encoding('gpt2')

torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
acc = torch.accelerator.current_accelerator()
backend = torch.distributed.get_default_backend_for_device(acc)
dist.init_process_group(backend)

rank = dist.get_rank()
device_id = rank % torch.accelerator.device_count()
device = f'cuda:{device_id}'

dataset = TokenDataset(np.memmap('tokens.bin', dtype=np.uint16, mode='r'),
                       max_seq_len)
ds_sampler = DistributedSampler(dataset, shuffle=True)
ds_loader = DataLoader(dataset, batch_size=batch_size, sampler=ds_sampler)
print('training data loaded')
vocab_size = Tokenizer.n_vocab

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len)
model = model.to(torch.bfloat16).to(device)
ddp_model = DDP(model, device_ids=[device_id])

optimizer = optim.AdamW(ddp_model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print('training started')

last_epoch = 0
if pathlib.Path('nanogpt_model.pt').exists():
    print('previous checkpoint found. loading....')
    chkp = torch.load('nanogpt_model.pt')
    model.load_state_dict(chkp['model'])
    optimizer.load_state_dict(chkp['optimizer'])
    last_epoch = chkp['epoch']
    
loss_counting = []
for epoch in range(last_epoch, n_epoch):
    ds_sampler.set_epoch(epoch)
    epoch_loss = 0
    epoch_total = 0
    step = 0
    for inputs, targets in ds_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _ = ddp_model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        epoch_loss += loss.item()
        epoch_total += 1
        step += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0:
            print(f'epoch {epoch} step {step}: loss={epoch_loss/epoch_total:.4f}')

    epoch_loss = epoch_loss / epoch_total
    if len(loss_counting) >= 50:
        loss_counting = loss_counting[1:]
    loss_counting.append(epoch_loss)
    print(f"Epoch {epoch}: loss={epoch_loss:.4f} avgloss={sum(loss_counting)/len(loss_counting):.4f}")
    if rank == 0:
        dist.barrier()
        print('saving checkpoint...')
        torch.save({
            'model': ddp_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, 'nanogpt_model.pt')

dist.destroy_process_group()

