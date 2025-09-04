import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)

    y[..., ::2] = -x[..., 1::2]
    y[..., 1::2] = x[..., 0::2]

    return y


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int = 32, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embedding = max_seq_len
        self.base = base

        # (head_dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        # (head_dim // 2) => (1, head_dim // 2, 1) => (B, head_dim // 2, 1)
        B, H, S, D = q.size()

        # (head_dim // 2) => (1, head_dim // 2, 1) => (B, head_dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(B, -1, 1)

        # (S,) => (B, S)  => (B, 1, S)
        position_ids = torch.arange(S, device=q.device).type_as(q)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        position_ids_expanded = position_ids[:, None, :]

        # Outer product
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, d_model)

        cos = emb.cos()
        sin = emb.sin()

        cos = cos.unsqueeze(1)  # (B, 1, S, d_model)
        sin = sin.unsqueeze(1)  # (B, 1, S, d_model)

        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
#     # (B, L, head_dim) => (B, 1, L, head_dim)
#     cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
#     sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension

#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
