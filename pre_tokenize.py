import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

# Define a directory to save the dataset
SAVE_DIR = "./data/tinystories"
VOCAB_SIZE = 16_000

# Load local files
train_file = os.path.join(SAVE_DIR, "TinyStoriesV2-GPT4-train.txt")
eval_file = os.path.join(SAVE_DIR, "TinyStoriesV2-GPT4-valid.txt")

with open(train_file, "r", encoding="utf-8") as f:
    train_texts = [line.strip() for line in f if line.strip()]

with open(eval_file, "r", encoding="utf-8") as f:
    eval_texts = [line.strip() for line in f if line.strip()]

print(f"Train split loaded from {train_file}")
print(f"Eval split loaded from {eval_file}")


# Helper function to encode corpus
def _encode_corpus(tokenizer, texts, eot_id: int | None):
    ids = []
    for t in tqdm(texts):
        # treat each non-empty line as one sample; append EOT if available
        enc = tokenizer.encode(t).ids
        if eot_id is not None:
            enc.append(eot_id)
        ids.extend(enc)
    return np.asarray(ids, dtype=np.uint32)


# Train a byte-level BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
tokenizer.normalizer = Sequence([NFC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<|endoftext|>",
    ],
)

tokenizer.train_from_iterator(train_texts, trainer=trainer)

# # Save the tokenizer files
tokenizer_json_path = os.path.join(SAVE_DIR, "tokenizer-bpe.json")
tokenizer.save(tokenizer_json_path)

tokenizer = Tokenizer.from_file(tokenizer_json_path)

# Encode and save train/eval splits as .npy files
eot_id = tokenizer.token_to_id("<|endoftext|>")
train_tokens = _encode_corpus(tokenizer, train_texts, eot_id)
eval_tokens = _encode_corpus(tokenizer, eval_texts, eot_id)
train_tokens_path = os.path.join(SAVE_DIR, "tinystories_train_tokens.npy")
eval_tokens_path = os.path.join(SAVE_DIR, "tinystories_eval_tokens.npy")
np.save(train_tokens_path, train_tokens)
np.save(eval_tokens_path, eval_tokens)
print(f"Train tokens: {len(train_tokens)} saved to {train_tokens_path}")
print(f"Eval tokens: {len(eval_tokens)} saved to {eval_tokens_path}")
print(f"Tokenizer JSON saved to {tokenizer_json_path}")
