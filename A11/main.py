import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import CharRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

with open('input.txt', 'r') as f:
    text = f.read()
print('training data loaded')

chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)

def get_batch(data, batch_size, seq_len):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s:s+seq_len] for s in starts])
    y = torch.stack([data[s+1:s+seq_len+1] for s in starts])
    return x.to(device), y.to(device)

d_embed = 128
d_hidden = 1024
batch_size = 64
seq_len = 128
n_steps = 5000
lr = 1e-3

model = CharRNN(vocab_size, d_embed, d_hidden).to(device)
# rnn = nn.RNN(d_embed, d_hidden, batch_first=True)
# gru = nn.GRU(d_embed, d_hidden, batch_first=True)
# lstm = nn.LSTM(d_embed, d_hidden, batch_first=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)

for step in range(n_steps):
    inputs, targets = get_batch(data, batch_size, seq_len)
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f'step {step}: loss={loss.item():.4f}')

torch.save({
    'model': model.state_dict()
}, 'nanogpt_model.pt')
print('model saved.')

