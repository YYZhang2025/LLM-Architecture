import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self._init_weight()

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        x = x + self.encoding[:seq_len, :].unsqueeze(0)

        if self.dropout:
            return self.dropout(x)

        return x

    def _init_weight(self):
        self.encoding.data.normal_(mean=0.0, std=0.02)
