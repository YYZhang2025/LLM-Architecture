from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    n_heads: int = 16
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 8

    max_seq_len: int = 512
    vocab_size: int = 16_000


@dataclass
class TrainConfig:
    device: torch.device = torch.device("cpu")

    epochs: int = 2
    micro_batch_size: int = 128
    gradient_accumulation_steps: int = 2

    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    weight_decay: float = 1e-2
    max_lr: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
