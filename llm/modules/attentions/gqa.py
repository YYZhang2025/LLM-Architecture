import torch
import torch.nn as nn

from llm.modules.position_encodings import RotaryPositionalEncoding


def repeat_kv(k, v, num_repeats):
    if num_repeats == 1:
        return k, v

    B, H, S, D = k.shape

    # (B, H, S, D) -> (B, H, 1, S, D)
    k = k[:, :, None, :, :].expand(B, H, num_repeats, S, D)
    v = v[:, :, None, :, :].expand(B, H, num_repeats, S, D)

    k, v = k.reshape(B, H * num_repeats, S, D), v.reshape(B, H * num_repeats, S, D)

    return k, v


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        n_query_heads: int = 8,
        n_kv_heads: int = 2,
        is_causal: bool = True,
        use_rope: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.num_repeats = n_query_heads // n_kv_heads
        self.is_causal = is_causal

        assert d_model % n_query_heads == 0, "d_model must be divisible by n_query_heads"

        self.head_dim = d_model // n_query_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, self.head_dim * n_query_heads)
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kv_heads)
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kv_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEncoding(
                head_dim=d_model // n_query_heads, **kwargs.get("rop_config", {})
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, S, D = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, S, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # HIGHLIGHT: The main different between MHA and GQA
        k, v = repeat_kv(k, v, self.num_repeats)

        if self.rope is not None:
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
