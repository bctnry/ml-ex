import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        d_in = linear.weight.shape[1]
        d_out = linear.weight.shape[0]
        # 0.01 is used to ensure A's contribution in the beginning is
        # near zero (so fine-tuning starts with the original performance
        # of the base model)
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))

    def forward(self, x):
        return self.linear(x) + (x @ self.A @ self.B)
    
