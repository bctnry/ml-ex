import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, d_embed, d_hidden):
        super().__init__()
        self.W_xh = nn.Linear(d_embed, d_hidden)
        self.W_hh = nn.Linear(d_hidden, d_hidden)
        self.W_hy = nn.Linear(d_hidden, d_embed)

    def forward(self, x_t, h_t):
        h_next = torch.tanh(self.W_hh(h_t) + self.W_xh(x_t))
        y_t = self.W_hy(h_next)
        return h_next, y_t

class CharRNN(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.cell = RNNCell(d_embed, d_hidden)
        self.head = nn.Linear(d_embed, vocab_size)

    def forward(self, token_ids, h_0=None):
        batch, seq_len = token_ids.shape
        x = self.embed(token_ids)

        if h_0 is None:
            h = torch.zeros(batch, self.cell.W_hh.out_features, device=x.device)
        else:
            h = h_0

        outputs = []
        for t in range(seq_len):
            h, y_t = self.cell(x[:, t, :], h)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        logits = self.head(y)
        return logits, h


