#!/usr/bin/env python3
"""
Run veRL SFT training.

Usage:
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --num-gpus 2
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --config config/latent_generation.yaml
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet trainer.total_epochs=1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_overrides(
    data: Path,
    num_gpus: int,
    sft_config: dict,
    checkpoint_dir: Path,
    wandb_enabled: bool,
) -> list[str]:
    """Build Hydra override list for veRL SFT trainer."""
    model = sft_config.get("model", "Qwen/Qwen3-4B")
    model_dtype = sft_config.get("model_dtype", "bf16")
    micro_batch = sft_config.get("micro_batch_size_per_gpu", 16)
    use_remove_padding = sft_config.get("use_remove_padding", True)
    use_liger = sft_config.get("use_liger", True)

    val_parquet = data.parent / "val.parquet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"model.partial_pretrain={model}",
        f"model.fsdp_config.model_dtype={model_dtype}",
        f"model.use_liger={str(use_liger).lower()}",
        f"data.train_files={data}",
        f"data.val_files={val_parquet if val_parquet.exists() else data}",
        f"data.prompt_key=prompt",
        f"data.response_key=response",
        f"data.max_length={sft_config.get('max_length', 256)}",
        f"data.train_batch_size={sft_config.get('train_batch_size', 64)}",
        f"data.micro_batch_size_per_gpu={micro_batch}",
        f"optim.lr={sft_config.get('lr', 1e-5)}",
        f"use_remove_padding={str(use_remove_padding).lower()}",
    ]

    epochs = sft_config.get("epochs")
    if epochs is not None:
        overrides.append(f"trainer.total_epochs={epochs}")

    if wandb_enabled:
        overrides.append("trainer.logger=['console','wandb']")
        project = os.environ.get("WANDB_PROJECT", "coarse-to-fine")
        overrides.append(f"trainer.project_name={project}")
    else:
        overrides.append("trainer.logger=['console']")

    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFT with veRL")
    parser.add_argument("--data", required=True, type=Path, help="Training parquet file")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (default: 1 or from config)")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML for SFT defaults and W&B")
    parser.add_argument("overrides", nargs="*", metavar="OVERRIDE", help="Extra Hydra overrides for veRL")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}", file=sys.stderr)
        return 1

    # Load config for defaults (or use empty dict)
    from src.utils.env import load_env, setup_wandb
    load_env()

    sft_config = {}
    wandb_enabled = False
    if args.config:
        if not args.config.exists():
            print(f"Error: Config not found: {args.config}", file=sys.stderr)
            return 1
        from src.config import load_config
        config = load_config(args.config)
        sft_config = config.get("sft", {})
        wandb_enabled = setup_wandb(config, step_name="sft")

    # CLI args override config
    num_gpus = args.num_gpus or int(sft_config.get("num_gpus", 1))
    checkpoint_dir = PROJECT_ROOT / sft_config.get("checkpoint_dir", "checkpoints/sft")

    overrides = build_overrides(
        data=args.data.resolve(),
        num_gpus=num_gpus,
        sft_config=sft_config,
        checkpoint_dir=checkpoint_dir,
        wandb_enabled=wandb_enabled,
    )
    overrides.extend(args.overrides)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "verl.trainer.fsdp_sft_trainer",
        *overrides,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
