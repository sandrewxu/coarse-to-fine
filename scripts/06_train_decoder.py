#!/usr/bin/env python3
"""
Pretrain the C2F (Coarse-to-Fine) joint model.

Reads config from config/experiments/latent_generation.yaml (c2f_training section),
loads training data (either flattened c2f_train.parquet or SFT train.parquet),
tokenizes into C2F format, and trains with HuggingFace Trainer + FSDP.

Tokenizer modes (set ``tokenizer`` in c2f_training config):
  - "space" (default): Train/load a word-level tokenizer where each
    space-separated word is exactly one token.  Guarantees perfect alignment
    between word count constraints and token positions.
  - "model": Use the BPE tokenizer from the init_from model checkpoint.

W&B integration (enabled via ``wandb.enabled: true`` in config):
  - Logs full experiment config as run metadata for run comparison
  - Logs dataset / model summary (size, param count, init source, vocab size)
  - Logs per-scale training loss (loss_z_4, loss_z_3, loss_z_2, loss_z_1, loss_text)
  - Standard HF Trainer metrics (loss, eval_loss, learning_rate, etc.)

Supports two dataset formats (set ``dataset_format`` in config):
  - "c2f":  flat word sequences from step 5 (c2f_train.parquet)
  - "sft":  prompt+response from step 3 (train.parquet, veRL format)

Usage:
    # Single GPU:
    python scripts/06_train_decoder.py --config config/experiments/latent_generation.yaml

    # Multi-GPU with accelerate + FSDP:
    accelerate launch --num_processes=4 scripts/06_train_decoder.py \
        --config config/experiments/latent_generation.yaml

    # Resume from checkpoint:
    python scripts/06_train_decoder.py \
        --config config/experiments/latent_generation.yaml \
        --resume-from checkpoints/decoder/checkpoint-500

Requires: pip install -e ".[c2f]" and a CUDA-capable environment.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training."""
    return os.environ.get("LOCAL_RANK", "0") == "0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain C2F model from latent generation data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiments" / "latent_generation.yaml",
        help="Path to experiment YAML with 'c2f_training' section",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config not found: {args.config}", file=sys.stderr)
        return 1

    # Load .env (secrets) and configure W&B before any training imports
    from src.utils.env import load_env, setup_wandb

    load_env()
    config = load_config(args.config)
    wandb_enabled = setup_wandb(config, step_name="c2f-pretrain")
    if wandb_enabled:
        print("W&B logging enabled for C2F pretraining")

    if "c2f_training" not in config:
        print("Error: Config missing 'c2f_training' section", file=sys.stderr)
        return 1

    c2f_config = config["c2f_training"]

    # Resolve dataset directory and format
    dataset_format = c2f_config.get("dataset_format", "c2f")
    dataset_dir = Path(c2f_config.get("dataset_dir", "data/local_generations"))
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    # Determine expected parquet filename based on format
    default_parquet = (
        "c2f_train.parquet" if dataset_format == "c2f" else "train.parquet"
    )
    parquet_filename = c2f_config.get("parquet_filename", default_parquet)
    parquet_path = dataset_dir / parquet_filename

    if not parquet_path.exists():
        print(f"Error: Training data not found: {parquet_path}", file=sys.stderr)
        if dataset_format == "c2f":
            print(
                "Run step 5 (05_generate_local.py) first to create the flattened training data.",
                file=sys.stderr,
            )
        else:
            print(
                "Run step 3 (03_verify_outputs.py) first to create the SFT dataset.",
                file=sys.stderr,
            )
        return 1

    from src.c2f_training.dataset import C2FDataset
    from src.c2f_training.train import C2FTrainer, build_training_args, load_c2f_model

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer_type = c2f_config.get("tokenizer", "space")
    tokenizer = None
    vocab_size = None

    if tokenizer_type == "space":
        from src.c2f_training.tokenizer import load_or_train_space_tokenizer

        tokenizer_dir = Path(
            c2f_config.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
        )
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = PROJECT_ROOT / tokenizer_dir

        tokenizer = load_or_train_space_tokenizer(
            tokenizer_dir=tokenizer_dir,
            data_dir=dataset_dir,
            dataset_format=dataset_format,
            parquet_filename=parquet_filename,
        )
        vocab_size = tokenizer.vocab_size
        print(f"  Space tokenizer ready (vocab_size={vocab_size})")
    else:
        # "model": use the tokenizer that ships with init_from checkpoint
        print("Using model BPE tokenizer")

    # ── Initialize W&B run (main process only) ──────────────────────────────
    # Manual init so the full experiment config is logged as run metadata.
    # HF Trainer's WandbCallback reuses this run instead of creating a new one.
    wandb_run = None
    if wandb_enabled and _is_main_process():
        import wandb

        wandb_run = wandb.init(
            name=c2f_config.get("run_name", "c2f-pretrain"),
            config={
                "step": "06-c2f-pretrain",
                "c2f_training": c2f_config,
                "scale_lengths": config.get("scale_lengths"),
                "word_count_constraints": config.get("word_count_constraints"),
                "text_word_count": config.get("text_word_count", 32),
                "dataset_format": dataset_format,
                "tokenizer": tokenizer_type,
                "vocab_size": vocab_size,
            },
        )

    # 1. Load model (pass vocab_size so random-init uses the right embedding size)
    print("Loading C2F model...")
    model = load_c2f_model(config, vocab_size=vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # 2. Build dataset
    tokenizer_source = c2f_config.get("init_from", "Qwen/Qwen3-4B")
    print(f"Building C2F dataset from {parquet_path} (format={dataset_format})...")
    full_dataset = C2FDataset(
        data_dir=str(dataset_dir),
        tokenizer_name_or_path=tokenizer_source,
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config["word_count_constraints"],
        text_word_count=config.get("text_word_count", 32),
        parquet_filename=parquet_filename,
        dataset_format=dataset_format,
        tokenizer=tokenizer,
    )
    print(f"  Dataset size: {len(full_dataset)}")

    # 3. Split into train/eval
    eval_split = c2f_config.get("eval_split", 0.05)
    splits = full_dataset.train_test_split(
        test_size=eval_split, seed=c2f_config.get("seed", 42)
    )
    train_size = len(splits["train"])
    eval_size = len(splits["test"])
    print(f"  Train: {train_size}, Eval: {eval_size}")

    # ── Log dataset / model metadata to W&B ─────────────────────────────────
    if wandb_run is not None:
        wandb_run.summary.update({
            "model/param_count": param_count,
            "model/init_from": c2f_config.get("init_from", "random"),
            "model/seq_len": full_dataset.seq_len,
            "model/vocab_size": vocab_size or "model-default",
            "dataset/total_size": len(full_dataset),
            "dataset/train_size": train_size,
            "dataset/eval_size": eval_size,
            "dataset/format": dataset_format,
            "tokenizer/type": tokenizer_type,
        })

    # 4. Build training args (W&B flag controls report_to)
    training_args = build_training_args(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)

    # 5. Create C2FTrainer (extends Trainer with per-scale loss logging)
    trainer = C2FTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        scale_lengths=config["scale_lengths"],
    )

    # 6. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 7. Save final model
    trainer.save_model()
    print(f"Model saved to: {training_args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
