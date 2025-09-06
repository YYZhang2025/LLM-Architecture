import math
import time

import torch
from torch import nn
from tqdm import tqdm

from llm.dataloaders import cycle_dataloader, get_dataloaders
from llm.utils import clear_cache, get_device, print_color


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        std = math.sqrt(2.0 / (module.weight.shape[0] + module.weight.shape[1]))
        nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)


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


def generate_samples(
    model,
    tokenizer,
    device,
    prompt: str = "Once upon a time in a land far, far away, ",
    max_length: int = 100,
    dtype: torch.dtype = torch.float16,
) -> str:
    is_training = model.training
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)  # (1, seq_len)
    ctx = (
        torch.autocast(device.type, dtype=torch.float16)
        if device.type != "cuda"
        else torch.autocast(device.type, dtype=torch.bfloat16)
    )

    generated_ids = []
    with torch.no_grad():
        for _ in range(max_length):
            with ctx:
                logits, _ = model(input_ids=input_ids)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # (1, 1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # (1, seq_len + 1)
            generated_ids.append(next_token_id.item())

            if next_token_id.item() == tokenizer.token_to_id("<|endoftext|>"):
                break

    generated_text = tokenizer.decode(generated_ids)
    model.train(is_training)

    print_color(f"Prompt: {prompt}", "yellow")
    print_color(f"Generated text: {generated_text}", "magenta")
    return prompt + generated_text


def eval_model(
    model: nn.Module,
    eval_dl: torch.utils.data.DataLoader,
    tokenizer,
    model_config,
    train_config,
    step,
    total_steps: int,
    run=None,
) -> float:
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(eval_dl, desc="Evaluating", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(train_config.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(train_config.device, non_blocking=True)
            labels = batch["labels"].to(train_config.device, non_blocking=True)

            with ctx:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, model_config.vocab_size),
                    labels.view(-1),
                )
            eval_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            del input_ids, attention_mask, labels, logits, loss
            clear_cache()

    eval_loss /= len(eval_dl)
    if run is not None:
        run.log({"eval/loss": eval_loss}, step=step + 1)

    print_color(f"Eval Loss ({step + 1} / {total_steps}): {eval_loss:.4f}", "cyan")
    generate_samples(model, tokenizer, train_config.device, max_length=100)

    return eval_loss


def train_model(model, tokenizer, model_config, train_config, run=None):
    ctx = (
        torch.autocast(train_config.device.type, dtype=torch.float16)
        if train_config.device.type != "cuda"
        else torch.autocast(train_config.device.type, dtype=torch.bfloat16)
    )

    train_dl, eval_dl = get_dataloaders(
        batch_size=train_config.micro_batch_size, seq_len=model_config.max_seq_len
    )
    total_steps = train_config.epochs * len(train_dl) // train_config.gradient_accumulation_steps
    print_color(f"Total training steps: {total_steps}", "green")
    train_dl = cycle_dataloader(train_dl)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.max_lr,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas,
    )

    total_start_time = time.time()
    for step in range(total_steps):
        model.train()
        batch_loss = 0.0
        start_time = time.time()
        for _ in range(train_config.gradient_accumulation_steps):
            batch = next(train_dl)
            input_ids = batch["input_ids"].to(train_config.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(train_config.device, non_blocking=True)
            labels = batch["labels"].to(train_config.device, non_blocking=True)

            with ctx:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, model_config.vocab_size),
                    labels.view(-1),
                )
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

            batch_loss += loss.item()
            del input_ids, attention_mask, labels, logits, loss
            clear_cache()

        lr = get_lr(train_config, step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        time_elapsed = time.time() - start_time
        if run is not None:
            run.log(
                {"train/loss": batch_loss, "train/elapsed_time": time_elapsed, "train/lr": lr},
                step=step + 1,
            )

        print(
            f"Step {step + 1}/{total_steps} | Loss: {batch_loss:.4f} | LR: {lr:.6f} | Time: {time_elapsed:.2f}s"
        )
        batch_loss = 0.0

        if (step + 1) % train_config.eval_steps == 0 or step == total_steps - 1:
            eval_model(model, eval_dl, tokenizer, model_config, train_config, step, total_steps, run)

    total_time = time.time() - total_start_time
    if run is not None:
        run.log({"train/total_time": total_time}, step=total_steps)
    print_color(f"Total training time: {total_time / 60:.2f} minutes", "green")
    print("Training completed.")
