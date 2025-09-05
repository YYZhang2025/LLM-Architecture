import torch
import torch.nn as nn

from llm.modules.position_encodings import RotaryPositionalEncoding


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, is_causal: bool = True, use_rope: bool = False, **kwargs):
        super().__init__()
        # Implementation of Flash Attention goes here

        self.d_model = d_model
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.use_rope = use_rope

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEncoding(head_dim=d_model // n_heads, **kwargs.get("rop_config", {}))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, S, D = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k)

        # Flash Attention implementation would go here
        raise NotImplementedError("Flash Attention is not yet implemented.")
