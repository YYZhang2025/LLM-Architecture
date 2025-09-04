import torch
import torch.nn as nn


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(base)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        x = x + self.pe[:, :seq_len, :]

        return x
