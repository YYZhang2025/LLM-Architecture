import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import ModelConfig
from llm.modules.attentions import MultiHeadedAttention
from llm.modules.ffn import MLP
from llm.modules.norms import LayerNorm
from llm.modules.position_encodings import SinePositionalEncoding


class Embedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = SinePositionalEncoding(d_model=config.d_model, max_len=config.max_seq_len)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        x = self.pos_enc(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()

        self.attn = MultiHeadedAttention(d_model=d_model, n_heads=n_heads, is_causal=True)
        self.ffn = MLP(d_model=d_model, d_ff=d_ff, act_fn=nn.GELU)

        self.norm1 = LayerNorm(dim=d_model)
        self.norm2 = LayerNorm(dim=d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        attn_out, attn_probs = self.attn(x, attention_mask=attention_mask)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_probs


class Baseline(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embedding = Embedding(config)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.apply(self._init_weight)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        return next_token

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x, attn_probs = layer(x, attention_mask=attention_mask)

        logits = F.linear(x, self.embedding.emb.weight)

        return logits, attn_probs

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(
                module.weight,
            )
        elif isinstance(module, LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
