import gc
import random

import numpy as np
import torch
from rich import print
from rich.pretty import pprint


def print_rich_dict(data: dict, title: str | None = None) -> None:
    """Pretty print dictionary with colors using rich."""
    if title:
        print("-------- ", title, " --------")
    pprint(data, expand_all=True)


def print_color(text: str, color: str = "red"):
    print(f"[{color}]{text}[/{color}]")


def get_num_parameters(model: torch.nn.Module) -> int:
    """Get the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_num_parameters(model: torch.nn.Module):
    num_params = get_num_parameters(model)
    print_color(f"Number of trainable parameters: {num_params:,}", color="green")


def seed_everything(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache():
    """Clear the PyTorch cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
