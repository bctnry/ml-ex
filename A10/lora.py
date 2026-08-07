import os
import pathlib
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NanoGPT
from lora_model import LoRALinear
from dataset3 import TokenDataset

# torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 ./lora.py

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 128
batch_size = 32


Tokenizer = tiktoken.get_encoding('gpt2')
vocab_size = Tokenizer.n_vocab

torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
acc = torch.accelerator.current_accelerator()
backend = torch.distributed.get_default_backend_for_device(acc)
dist.init_process_group(backend)

rank = dist.get_rank()
device_id = rank % torch.accelerator.device_count()
device = f'cuda:{device_id}'

dataset = TokenDataset(np.memmap('tinyshakespeare_tokens.bin', dtype=np.uint16, mode='r'),
                       max_seq_len)
ds_sampler = DistributedSampler(dataset, shuffle=True)
ds_loader = DataLoader(dataset, batch_size=batch_size, sampler=ds_sampler)
print('training data loaded')

# create model -> load base model -> freeze model params
#              -> add LoRA -> load LoRA state
#              -> ddp
model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)
chkp = torch.load('nanogpt_model.pt', map_location=device)
model_state_dict = chkp['model']
model.load_state_dict(model_state_dict)

for p in model.parameters():
    p.requires_grad = False
    
for block in model.blocks:
    block.attn.Wq = LoRALinear(block.attn.Wq, rank=8)
    block.attn.Wv = LoRALinear(block.attn.Wv, rank=8)

if pathlib.Path('lora_tinyshakespeare.pt').exists():
    chkp = torch.load('lora_tinyshakespeare.pt')
    model.load_state_dict(chkp['lora'], strict=False)

model = model.to(device)
    
ddp_model = DDP(model, device_ids=[device_id])

lr_peak = 1e-4
weight_decay = 1e-2

optimizer = optim.AdamW(
    [p for p in ddp_model.parameters() if p.requires_grad],
    lr=lr_peak,
    weight_decay=weight_decay
)

warmup_steps = 500
total_steps = 5000

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if chkp.get('lora_optimizer') is not None:
    optimizer.load_state_dict(chkp['lora_optimizer'])

print('lora fine-tuning started')

if pathlib.Path('lora_tinystories.pt').exists():
    last_step = chkp['step']
else:
    last_step = 0
step = last_step
ds_sampler.set_epoch(0)
epoch_loss = 0
epoch_total = 0
for inputs, targets in ds_loader:
    print('.', end='', flush=True)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _, _ = ddp_model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    epoch_loss += loss
    epoch_total += 1
    step += 1
    if step > total_steps:
        break
    if step % 50 == 0:
        print(f'step {step}: loss={epoch_loss/epoch_total:.4f}')
    if step % 200 == 0:
        if rank == 0:
            dist.barrier()
            print('saving checkpoint...', flush=True)
            torch.save({
                'lora': {k: v for k, v in ddp_model.module.state_dict().items() if 'lora' in k or 'A' in k or 'B' in k},
                'lora_optimizer': optimizer.state_dict(),
                'step': step,
            }, 'lora_tinystories.pt')
        

if rank == 0:
    dist.barrier()
    print('saving checkpoint...')
    torch.save({
        'lora': {k: v for k, v in ddp_model.module.state_dict().items() if 'lora' in k or 'A' in k or 'B' in k},
        'lora_optimizer': optimizer.state_dict(),
        'step': step,
    }, 'lora_tinyshakespeare.pt')
    
