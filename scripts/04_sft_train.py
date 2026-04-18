#!/usr/bin/env python3
"""Step 4 — SFT q_φ (Qwen3-4B) on verified batch outputs.

Usage:
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --config config/latent_generation.yaml
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --num-gpus 2 --epochs 3

    # Multi-GPU with FSDP:
    accelerate launch --num_processes=2 scripts/04_sft_train.py \
        --data data/sft_dataset/train.parquet --config config/latent_generation.yaml

The training logic lives in ``src/sft/train.py``; this script is a thin CLI.
"""

import argparse
import sys
from pathlib import Path

from src.common.env import load_env, setup_wandb
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT
from src.config import load_config
from src.sft.train import train_sft

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training with HuggingFace Trainer")
    parser.add_argument(
        "--data", required=True, type=Path, help="Training parquet (columns: prompt, response)"
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-device batch size"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory"
    )
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    load_env()

    config: dict = {}
    wandb_enabled = False
    if args.config is not None:
        if not args.config.exists():
            log.error("Config not found: %s", args.config)
            return 1
        config = load_config(args.config)
        wandb_enabled = setup_wandb(config, step_name="sft")

    cli_overrides = {
        "epochs": args.epochs,
        "lr": args.lr,
        "per_device_batch_size": args.batch_size,
        "checkpoint_dir": args.checkpoint_dir,
        "num_gpus": args.num_gpus,
    }

    return train_sft(
        config,
        PROJECT_ROOT,
        data_path=args.data,
        wandb_enabled=wandb_enabled,
        resume_from=args.resume_from,
        cli_overrides=cli_overrides,
    )


if __name__ == "__main__":
    sys.exit(main())
