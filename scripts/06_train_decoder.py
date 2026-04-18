#!/usr/bin/env python3
"""
Train the C2F (Coarse-to-Fine) joint model.

Usage:
    # Minimal — random init, defaults for everything:
    python scripts/06_train_decoder.py --data data/sft_dataset/train.parquet

    # Init from pretrained, with config for full control:
    python scripts/06_train_decoder.py \
        --data data/sft_dataset/train.parquet \
        --init-from Qwen/Qwen3-4B \
        --config config/latent_generation.yaml

    # Override training params from CLI:
    python scripts/06_train_decoder.py \
        --data data/sft_dataset/train.parquet \
        --epochs 5 --lr 1e-4 --batch-size 16 \
        --checkpoint-dir checkpoints/decoder/run2

    # Multi-GPU with accelerate:
    accelerate launch --num_processes=4 scripts/06_train_decoder.py \
        --data data/sft_dataset/train.parquet \
        --config config/latent_generation.yaml

    # Resume from checkpoint:
    python scripts/06_train_decoder.py \
        --data data/sft_dataset/train.parquet \
        --config config/latent_generation.yaml \
        --resume-from checkpoints/decoder/checkpoint-500
"""

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.common.logging import get_logger

log = get_logger(__name__)

from src.common.paths import PROJECT_ROOT


def detect_dataset_format(parquet_path: Path) -> str:
    """Auto-detect dataset format from parquet columns."""
    columns = pq.read_schema(str(parquet_path)).names
    if "text" in columns:
        return "c2f"
    if "prompt" in columns and "response" in columns:
        return "sft"
    raise ValueError(
        f"Cannot detect format from columns {columns}. "
        "Expected 'text' (c2f) or 'prompt'+'response' (sft)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train C2F decoder model")
    # Essential inputs
    parser.add_argument("--data", required=True, type=Path, help="Training parquet file")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Model init source: 'random' or checkpoint/HF path (default: from config or 'random')",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Experiment YAML for defaults and W&B"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from checkpoint directory"
    )
    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-device batch size"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="Override checkpoint output directory"
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default=None,
        choices=["block", "causal"],
        help="Attention mask: 'block' (C2F prefix) or 'causal' (standard autoregressive)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        log.error(f"Data file not found: {args.data}")
        return 1

    # Load config for defaults (or use empty)
    config = {"scale_lengths": [2, 4, 8, 16, 32], "c2f_training": {}}
    wandb_enabled = False

    if args.config:
        if not args.config.exists():
            log.error(f"Config not found: {args.config}")
            return 1
        from src.common.env import load_env, setup_wandb
        from src.config import load_config

        load_env()
        config = load_config(args.config)
        wandb_enabled = setup_wandb(config, step_name="c2f-pretrain")

    c2f_cfg = config["c2f_training"]

    # CLI overrides
    if args.init_from is not None:
        c2f_cfg["init_from"] = args.init_from
    if args.epochs is not None:
        c2f_cfg["epochs"] = args.epochs
    if args.lr is not None:
        c2f_cfg["lr"] = args.lr
    if args.batch_size is not None:
        c2f_cfg["per_device_batch_size"] = args.batch_size
    if args.checkpoint_dir is not None:
        c2f_cfg["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.mask_type is not None:
        c2f_cfg["mask_type"] = args.mask_type

    # Detect dataset format and resolve paths
    data_path = args.data.resolve()
    dataset_dir = data_path.parent
    parquet_filename = data_path.name
    dataset_format = detect_dataset_format(data_path)
    log.info(f"Dataset: {data_path} (format={dataset_format})")

    # Tokenizer
    from src.c2f_model.training.dataset import C2FDataset
    from src.c2f_model.training.train import C2FTrainer, build_training_args, load_c2f_model

    tokenizer_type = c2f_cfg.get("tokenizer", "space")
    tokenizer = None
    vocab_size = None

    if tokenizer_type == "space":
        from src.c2f_model.training.tokenizer import load_or_train_space_tokenizer

        tokenizer_dir = Path(c2f_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer"))
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = PROJECT_ROOT / tokenizer_dir
        tokenizer = load_or_train_space_tokenizer(
            tokenizer_dir=tokenizer_dir,
            data_dir=dataset_dir,
            dataset_format=dataset_format,
            parquet_filename=parquet_filename,
        )
        vocab_size = tokenizer.vocab_size
        log.info(f"  Space tokenizer ready (vocab_size={vocab_size})")

    # Init W&B (main process only)
    wandb_run = None
    if wandb_enabled and os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb

        wandb_run = wandb.init(
            name=c2f_cfg.get("run_name", "c2f-pretrain"),
            config={
                "step": "06-c2f-pretrain",
                "c2f_training": c2f_cfg,
                "scale_lengths": config.get("scale_lengths"),
            },
        )

    # Load model
    log.info(f"Loading C2F model (init_from={c2f_cfg.get('init_from', 'random')})...")
    model = load_c2f_model(config, vocab_size=vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {param_count:,}")

    # Build dataset
    log.info("Building dataset...")
    full_dataset = C2FDataset(
        data_dir=str(dataset_dir),
        tokenizer_name_or_path=c2f_cfg.get("init_from", "Qwen/Qwen3-4B"),
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config.get("word_count_constraints", {}),
        text_word_count=config.get("text_word_count", 32),
        parquet_filename=parquet_filename,
        dataset_format=dataset_format,
        tokenizer=tokenizer,
    )

    # Split
    eval_split = c2f_cfg.get("eval_split", 0.05)
    splits = full_dataset.train_test_split(test_size=eval_split, seed=c2f_cfg.get("seed", 42))
    log.info(f"  Train: {len(splits['train'])}, Eval: {len(splits['test'])}")

    # Train
    training_args = build_training_args(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
    trainer = C2FTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        scale_lengths=config["scale_lengths"],
        mask_type=c2f_cfg.get("mask_type", "block"),
    )

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    log.info(f"Model saved to: {training_args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
