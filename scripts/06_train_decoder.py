#!/usr/bin/env python3
"""
Pretrain the C2F (Coarse-to-Fine) joint model.

Reads config from config/experiments/latent_generation.yaml (c2f_training section),
loads step 5 generation outputs (flattened c2f_train.parquet), tokenizes into C2F
format, and trains with HuggingFace Trainer + FSDP.

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
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # Resolve dataset directory
    dataset_dir = Path(c2f_config.get("dataset_dir", "data/local_generations"))
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    c2f_parquet = dataset_dir / "c2f_train.parquet"
    if not c2f_parquet.exists():
        print(f"Error: C2F training data not found: {c2f_parquet}", file=sys.stderr)
        print(
            "Run step 5 (05_generate_local.py) first to create the flattened training data.",
            file=sys.stderr,
        )
        return 1

    from transformers import Trainer

    from src.c2f_training.dataset import C2FDataset
    from src.c2f_training.train import build_training_args, load_c2f_model

    # 1. Load model
    print("Loading C2F model...")
    model = load_c2f_model(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Build dataset
    tokenizer_source = c2f_config.get("init_from", "Qwen/Qwen3-4B")
    print(f"Building C2F dataset from {c2f_parquet}...")
    full_dataset = C2FDataset(
        data_dir=str(dataset_dir),
        tokenizer_name_or_path=tokenizer_source,
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config["word_count_constraints"],
        text_word_count=config.get("text_word_count", 32),
    )
    print(f"  Dataset size: {len(full_dataset)}")

    # 3. Split into train/eval
    eval_split = c2f_config.get("eval_split", 0.05)
    splits = full_dataset.train_test_split(
        test_size=eval_split, seed=c2f_config.get("seed", 42)
    )
    print(f"  Train: {len(splits['train'])}, Eval: {len(splits['test'])}")

    # 4. Build training args (W&B flag controls report_to)
    training_args = build_training_args(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
    )

    # 6. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 7. Save final model
    trainer.save_model()
    print(f"Model saved to: {training_args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
