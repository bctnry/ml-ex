import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import CharRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

with open('input.txt', 'r') as f:
    s = f.read()
print('training data loaded')

chars = sorted(set(s))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

d_embed = 128
d_hidden = 1024
batch_size = 64
seq_len = 128

model = CharRNN(vocab_size, d_embed, d_hidden)
chkp = torch.load('nanogpt_model.pt', map_location=device)
model.load_state_dict(chkp['model'])
model = model.to(device)

@torch.no_grad()
def generate(model, prompt, n_tokens, temperature=1.0):
    model.eval()
    token_ids = torch.tensor([[char_to_id[c] for c in prompt]]).to(device)

    logits, h = model(token_ids)
    next_logits = logits[0, -1, :] / temperature
    next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
    print(prompt + id_to_char[next_token.item()], end='', flush=True)

    for _ in range(n_tokens):
        # RNN consumes uses one (1) token at a time
        embed = model.embed(next_token[None, :])
        h, y_t = model.cell(embed[:, 0, :], h)
        next_logits = model.head(y_t)[0] / temperature
        next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
        print(id_to_char[next_token.item()], end='', flush=True)
    print()

generate(model, 'JULIET:', 1000, temperature=0.8)


