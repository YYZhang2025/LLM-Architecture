from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, act_fn: Callable = nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act_fn = act_fn()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)

        return x
