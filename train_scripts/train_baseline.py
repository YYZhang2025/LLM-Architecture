import os
import warnings
from dataclasses import asdict
from pathlib import Path

import dotenv
import torch
from tokenizers import Tokenizer

import wandb
from llm.config import TrainConfig
from llm.models.baseline import Baseline, ModelConfig
from llm.train_engine import init_weights, train_model
from llm.utils import (
    get_device,
    get_num_parameters,
    print_color,
    print_num_parameters,
    print_rich_dict,
    seed_everything,
)

warnings.filterwarnings("ignore")
TOKENIZER_JSON_PATH = "./data/tinystories/tokenizer-bpe.json"


if __name__ == "__main__":
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything()

    model_config = ModelConfig()
    train_config = TrainConfig()
    train_config.device = get_device()
    # Use small batch size for CPU/GPU for debug
    if "cuda" not in train_config.device.type:
        train_config.micro_batch_size = 2

    model = Baseline(model_config).to(train_config.device)
    model.apply(init_weights)
    tokenizer = Tokenizer.from_file(TOKENIZER_JSON_PATH)

    # INITIALIZE WANDB
    run = None
    if os.getenv("WANDB_API_KEY") is not None:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "version": "baseline",
                "num_params": get_num_parameters(model),
                "train_config": asdict(train_config),
                "model_cfg": asdict(model_config),
            },
            name=model_config.model_name,
        )

    print_rich_dict(asdict(model_config), title="Model Config")
    print_rich_dict(asdict(train_config), title="Train Config")
    print_num_parameters(model)

    train_model(model, tokenizer, model_config, train_config, run)

    # --------------
    # SAVE THE MODEL
    # --------------
    Path.mkdir(Path("checkpoints"), exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{model_config.model_name}-model.pth")
    print_color(f"Model saved to checkpoints/{model_config.model_name}-model.pth", "green")
