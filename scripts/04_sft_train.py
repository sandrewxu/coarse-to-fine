#!/usr/bin/env python3
"""
Run Supervised Fine-Tuning with veRL.

Reads SFT options from config/experiments/latent_generation.yaml (model, num_gpus, dataset_dir,
checkpoint_dir), builds veRL Hydra overrides, and launches the veRL SFT trainer via
torchrun with nproc_per_node = num_gpus. Checkpoints are saved to checkpoints/sft.

Usage:
    python scripts/04_sft_train.py [--config CONFIG] [OVERRIDES...]

    Overrides are passed to the veRL trainer (Hydra-style), e.g.:
        trainer.resume_mode=disable
        trainer.total_epochs=1
        data.micro_batch_size_per_gpu=16

Requires: pip install -e ".[sft]" and a veRL-compatible environment (see veRL docs).
Dataset: data/sft_dataset/train.parquet (created by verification step 03 with sft.dataset_dir).
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SFT with veRL (config: model, num_gpus, dataset_dir, checkpoint_dir)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiments" / "latent_generation.yaml",
        help="Path to experiment YAML with 'sft' section",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="OVERRIDE",
        help="Extra Hydra overrides for veRL (e.g. trainer.resume_mode=disable)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config not found: {args.config}", file=sys.stderr)
        return 1

    # Load .env (secrets) and configure W&B before any training imports
    from src.utils.env import load_env, setup_wandb

    load_env()
    config = load_config(args.config)
    wandb_enabled = setup_wandb(config, step_name="sft")
    if wandb_enabled:
        print("W&B logging enabled for SFT")

    sft_config = config.get("sft") or {}
    num_gpus = int(sft_config.get("num_gpus", 1))
    if num_gpus < 1 or num_gpus > 4:
        print("Error: sft.num_gpus must be 1, 2, 3, or 4", file=sys.stderr)
        return 1

    from src.sft.train import build_verl_sft_overrides, get_verl_sft_entrypoint

    overrides = build_verl_sft_overrides(sft_config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
    overrides.extend(args.overrides)
    entrypoint = get_verl_sft_entrypoint()

    train_parquet = (PROJECT_ROOT / sft_config.get("dataset_dir", "data/sft_dataset")) / "train.parquet"
    if not train_parquet.exists():
        print(f"Error: SFT dataset not found: {train_parquet}", file=sys.stderr)
        print("Run the verification step (03) first so that SFT data is written to data/sft_dataset.", file=sys.stderr)
        return 1

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m",
        entrypoint,
        *overrides,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
