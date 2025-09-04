from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    device: torch.device = torch.device("cpu")

    epochs: int = 2
    micro_batch_size: int = 128
    gradient_accumulation_steps: int = 2
    eval_steps: int = 100

    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    weight_decay: float = 1e-2
    max_lr: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
