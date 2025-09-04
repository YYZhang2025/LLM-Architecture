import os
import warnings
from dataclasses import asdict
from pathlib import Path

import dotenv
import torch
from tokenizers import Tokenizer

from llm.config import TrainConfig
from llm.models.baseline import Baseline, ModelConfig
from llm.train_engine import train_model
from llm.utils import (
    get_device,
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

    model_config = ModelConfig()

    train_config = TrainConfig()
    train_config.device = get_device()

    seed_everything(42)

    # INITIALIZE WANDB
    run = None
    # if os.getenv("WANDB_API_KEY") is not None:
    #     api_key = os.getenv("WANDB_API_KEY")
    #     wandb.login(key=api_key)
    #     run = wandb.init(
    #         project=os.getenv("WANDB_PROJECT"),
    #         entity=os.getenv("WANDB_ENTITY"),
    #         config={
    #             "version": "baseline",
    #             "train_config": asdict(train_config),
    #             "model_cfg": asdict(model_config),
    #         },
    #         name="baseline",
    #     )

    print_rich_dict(asdict(model_config), title="Model Config")
    print_rich_dict(asdict(train_config), title="Train Config")

    model = Baseline(model_config).to(train_config.device)
    print_num_parameters(model)
    tokenizer = Tokenizer.from_file(TOKENIZER_JSON_PATH)

    train_model(model, tokenizer, model_config, train_config, run)

    # --------------
    # SAVE THE MODEL
    # --------------
    Path.mkdir(Path("checkpoints"), exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baseline-model.pth")
    print_color("Model saved to checkpoints/baseline-model.pth", "green")
