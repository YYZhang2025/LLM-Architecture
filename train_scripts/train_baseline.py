import os
from dataclasses import asdict

import dotenv
import torch

from llm.dataloaders import cycle_dataloader, get_dataloaders
from llm.models.baseline import Baseline, ModelConfig
from llm.utils import get_device, get_num_parameters, print_color, print_rich_dict

from .train_config import TrainConfig


def train(model_config, train_config):
    ctx = (
        torch.autocast(train_config.device.type, dtype=torch.float16)
        if train_config.device.type != "cuda"
        else torch.autocast(train_config.device.type, dtype=torch.bfloat16)
    )

    model = Baseline(model_config).to(train_cfg.device)
    print_color(f"Model initialized. Number of parameters: {get_num_parameters(model)}", color="green")
    train_dl, eval_dl = get_dataloaders()
    train_dl = cycle_dataloader(train_dl)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas
    )

    for step in range(train_cfg.total_steps):
        model.train()
        for _ in range(train_cfg.gradient_accumulation_steps):
            batch = next(train_dl)
            input_ids = batch["input_ids"].to(train_cfg.device)
            attention_mask = batch["attention_mask"].to(train_cfg.device)
            labels = batch["labels"].to(train_cfg.device)
            labels = labels[1:].contiguous().clone()

            with ctx:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (
                    torch.nn.functional.cross_entropy(
                        logits.view(-1, model_config.vocab_size)[::-1, :],
                        labels.view(-1),
                    )
                    / train_cfg.gradient_accumulation_steps
                )

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{train_cfg.total_steps}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_config = ModelConfig()

    train_cfg = TrainConfig()
    train_cfg.device = get_device()

    print_rich_dict(asdict(model_config), title="Model Config")
    print_rich_dict(asdict(train_cfg), title="Train Config")

    train(model_config, train_cfg)
