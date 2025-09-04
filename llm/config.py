from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    n_heads: int = 12
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 8

    max_seq_len: int = 1024
    vocab_size: int = 16_000


@dataclass
class TrainConfig:
    device: torch.device = torch.device("cpu")

    weight_decay: float = 1e-2

    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    total_steps: int = 1000

    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    max_lr: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
