import torch
import torch.nn as nn

from llm.modules.position_encodings import RotaryPositionalEncoding


class MultiHeadedAttention(nn.Module):
    def __init__(
        self, d_model: int = 2048, n_heads: int = 8, is_causal: bool = True, use_rope: bool = False, **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEncoding(head_dim=d_model // n_heads, **kwargs.get("rop_config", {}))

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, S, D = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling

        if self.is_causal:
            causal_mask = torch.tril(torch.ones(1, 1, S, S, device=x.device).bool())
            if attention_mask is not None:
                # TODO: Since our encoding method has no <pad> tokens, so the attention mask is None
                # Need to check whether this is correct when the attention is not all 1
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                causal_mask = causal_mask & attention_mask
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))

        attn_probs = attn_weights.softmax(dim=-1)
        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        return out, attn_probs
