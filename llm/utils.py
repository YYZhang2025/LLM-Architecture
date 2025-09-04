import gc
import math
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


def generate_samples(
    model,
    tokenizer,
    device,
    prompt: str,
    max_length: int = 100,
) -> str:
    is_training = model.training
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids=input_ids)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # (1, 1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # (1, seq_len + 1)

            if next_token_id.item() == tokenizer.token_to_id("<|endoftext|>"):
                break

    generated_text = tokenizer.decode(input_ids.squeeze().tolist())
    model.train(is_training)
    return prompt + generated_text


def get_lr(train_config, cur_step: int, total_steps: int) -> float:
    if cur_step < train_config.warmup_steps:
        return train_config.max_lr * (cur_step + 1) / train_config.warmup_steps

    if cur_step > total_steps:
        return train_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (cur_step - train_config.warmup_steps) / (total_steps - train_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0

    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)
