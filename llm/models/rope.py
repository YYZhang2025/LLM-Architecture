from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.modules.attentions import MultiHeadedAttention
from llm.modules.ffn import MLP
from llm.modules.norms import RMSNorm


@dataclass
class ModelConfig:
    model_name: str = "RoPE"
    n_heads = 16
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 4

    max_seq_len: int = 1028
    vocab_size: int = 10_000


class Embedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.emb = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.attn = MultiHeadedAttention(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            is_causal=True,
            use_rope=True,
            rop_config={"max_seq_len": model_config.max_seq_len},
        )
        self.ffn = MLP(d_model=model_config.d_model, d_ff=model_config.d_ff, act_fn=nn.GELU)

        self.norm1 = RMSNorm(dim=model_config.d_model)
        self.norm2 = RMSNorm(dim=model_config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        attn_out, attn_probs = self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x, attn_probs


class RoPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    model_config=config,
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
