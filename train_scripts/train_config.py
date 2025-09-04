from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    device: torch.device = torch.device("cpu")
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2

    micor_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    total_steps: int = 500

    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
