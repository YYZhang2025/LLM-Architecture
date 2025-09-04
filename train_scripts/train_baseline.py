import math
import os
import time
from dataclasses import asdict

import dotenv
import torch
from tokenizers import Tokenizer

import wandb
from llm.config import TrainConfig
from llm.dataloaders import cycle_dataloader, get_dataloaders
from llm.models.baseline import Baseline, ModelConfig
from llm.utils import clear_cache, get_device, get_num_parameters, print_color, print_rich_dict

TOKENIZER_JSON_PATH = "./data/tinystories/tokenizer-bpe.json"


def get_lr(train_config: TrainConfig, cur_step: int) -> float:
    if cur_step < train_config.warmup_steps:
        return train_config.max_lr * (cur_step + 1) / train_config.warmup_steps

    if cur_step > train_config.total_steps:
        return train_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (cur_step - train_config.warmup_steps) / (
        train_config.total_steps - train_config.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)


def train(model_config, train_config, run=None):
    ctx = (
        torch.autocast(train_config.device.type, dtype=torch.float16)
        if train_config.device.type != "cuda"
        else torch.autocast(train_config.device.type, dtype=torch.bfloat16)
    )

    model = Baseline(model_config).to(train_cfg.device)
    print_color(f"Model initialized. Number of parameters: {get_num_parameters(model)}", color="green")
    train_dl, eval_dl = get_dataloaders(
        batch_size=train_cfg.micro_batch_size, seq_len=model_config.max_seq_len
    )
    train_dl = cycle_dataloader(train_dl)
    tokenizer = Tokenizer.from_file(TOKENIZER_JSON_PATH)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas
    )

    for step in range(train_cfg.total_steps):
        model.train()
        batch_loss = 0.0
        start_time = time.time()
        for _ in range(train_cfg.gradient_accumulation_steps):
            batch = next(train_dl)
            input_ids = batch["input_ids"].to(train_cfg.device)
            attention_mask = batch["attention_mask"].to(train_cfg.device)
            labels = batch["labels"].to(train_cfg.device)

            with ctx:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, model_config.vocab_size),
                    labels.view(-1),
                )
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        lr = get_lr(train_cfg, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        time_elapsed = time.time() - start_time
        if run is not None:
            run.log(
                {"train/loss": batch_loss, "train/elapsed_time": time_elapsed, "train/lr": lr}, step=step + 1
            )

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{train_cfg.total_steps}, Loss: {batch_loss:.4f}")
            sample_input_id = input_ids[:1, :]
            sample_logits, _ = model(input_ids=sample_input_id, attention_mask=None)
            sample_next_token = torch.argmax(sample_logits, dim=-1)

            sample_input_id = sample_input_id.squeeze(0)
            sample_next_token = sample_next_token.squeeze(0)
            print("Sample input ids:", tokenizer.decode(sample_input_id.cpu().numpy()))
            print("Sample output", tokenizer.decode(sample_next_token.cpu().numpy()))

        batch_loss = 0.0
        clear_cache()

    print("Training completed.")


if __name__ == "__main__":
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_config = ModelConfig()

    train_cfg = TrainConfig()
    train_cfg.device = get_device()
    train_cfg.total_steps = 1000
    train_cfg.micro_batch_size = 8
    train_cfg.gradient_accumulation_steps = 4
    model_config.max_seq_len = 1024

    print_rich_dict(asdict(model_config), title="Model Config")
    print_rich_dict(asdict(train_cfg), title="Train Config")

    # INITIALIZE WANDB
    run = None
    if os.getenv("WANDB_API_KEY") is not None:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            config={"version": "baseline", "train_cfg": asdict(train_cfg), "model_cfg": asdict(model_config)},
            name="baseline",
        )

    train(model_config, train_cfg, run)
