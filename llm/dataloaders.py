import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TinyStoriesDataset(Dataset):
    def __init__(self, npy_file, seq_len=512, stride=None):
        self.tokens = np.load(npy_file, mmap_mode="r")
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.num = (len(self.tokens) - seq_len) // self.stride

    def __len__(self):
        return max(0, self.num)

    def __getitem__(self, i):
        s = i * self.stride
        x = torch.as_tensor(self.tokens[s : s + self.seq_len], dtype=torch.long)

        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


def get_dataloaders(train_npy_path, eval_npy_path, batch_size=8, seq_len=512, num_workers=4, pin_memory=True):
    train_ds = TinyStoriesDataset(train_npy_path, seq_len=seq_len)
    eval_ds = TinyStoriesDataset(eval_npy_path, seq_len=seq_len)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_dl, eval_dl


if __name__ == "__main__":
    train_path = "./data/tinystories/tinystories_train_tokens.npy"
    eval_path = "./data/tinystories/tinystories_eval_tokens.npy"
    train_dl, eval_dl = get_dataloaders(train_path, eval_path, batch_size=2)
    batch = next(iter(train_dl))

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)

    print(batch["input_ids"])
