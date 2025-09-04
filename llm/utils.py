import gc

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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_parameters(model: torch.nn.Module) -> int:
    """Get the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clear_cache():
    """Clear the PyTorch cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
