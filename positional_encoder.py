import torch
import torch.nn as nn
import math

# Following the official Pytorch tutorial
# From https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoder(nn.Module):
    def __init__(self, d_model,
                 dropout=0.1,
                 max_seq_len=5000):
        
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0 / d_model))
        pe       = torch.zeros(max_seq_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.dropout(x + self.pe[:x.size(0)])
