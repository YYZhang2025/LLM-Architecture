import torch.nn as nn
from flash_attn import flash_attn_func

from llm.modules.position_encodings import RotaryPositionalEncoding


class FlashMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, is_causal: bool = True, use_rope: bool = False, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEncoding(head_dim=d_model // n_heads, **kwargs.get("rop_config", {}))

    def forward(self, x, attention_mask=None):
        assert attention_mask is None, "Right Now we only support attention_mask=None for FlashMHA"
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.hd)
        k = k.view(B, T, self.nh, self.hd)
        v = v.view(B, T, self.nh, self.hd)

        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k)

        out = flash_attn_func(
            q,
            k,
            v,
            causal=self.is_causal,
        )
        out = out.reshape(B, T, C)
        return self.proj(out), None
