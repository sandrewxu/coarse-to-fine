#!/usr/bin/env python3
"""
Step 7: ELBO Optimisation — GRPO on q_φ / Supervised on p_θ / Joint.

Phases:
  sft   — Phase A: GRPO on q_φ (SFT model), p_θ (C2F) frozen.
  c2f   — Phase B: Supervised fine-tuning of p_θ (C2F), q_φ (SFT) frozen.
  both  — Phase A then Phase B sequentially.
  joint — Simultaneous SFT + C2F training (placeholder for custom veRL mod).

Usage:
    python scripts/07_rl_train.py --phase sft   --config config/latent_generation.yaml
    python scripts/07_rl_train.py --phase c2f   --config config/latent_generation.yaml
    python scripts/07_rl_train.py --phase both  --config config/latent_generation.yaml
    python scripts/07_rl_train.py --phase joint --config config/latent_generation.yaml

    # Override RL config values:
    python scripts/07_rl_train.py --phase sft --config config/latent_generation.yaml \
        rl.sft_rl.epochs=1 rl.sft_rl.train_batch_size=8

    # Pass-through Hydra overrides to veRL (Phase A only):
    python scripts/07_rl_train.py --phase sft --config config/latent_generation.yaml \
        trainer.total_epochs=2
"""

import argparse
import os
import sys
from pathlib import Path

from src.common.logging import get_logger

log = get_logger(__name__)

from src.common.paths import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 7: ELBO optimisation")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["sft", "c2f", "both", "joint"],
        help="Training phase to run",
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="OVERRIDE",
        help="rl.* overrides (applied to config) or Hydra overrides (Phase A)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        log.error(f"Config not found: {args.config}")
        return 1

    # Workaround: prevent uv from re-syncing inside torchrun subprocesses
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    _venv = os.environ.get("VIRTUAL_ENV") or str(Path(sys.executable).parent.parent)
    os.environ["UV_NO_SYNC"] = "1"
    os.environ["UV_PROJECT_ENVIRONMENT"] = _venv

    from src.common.env import load_env, setup_wandb
    from src.config import load_config

    load_env()
    config = load_config(args.config)
    wandb_enabled = setup_wandb(config, step_name=f"rl-{args.phase}")

    # Apply dot-path overrides (rl.* → config, others → veRL pass-through)
    from src.rl.train import apply_overrides

    config, verl_overrides = apply_overrides(config, args.overrides)

    # Dispatch
    from src.rl.train import run_c2f_finetune, run_joint, run_sft_rl

    if args.phase in ("sft", "both"):
        log.info("=" * 60)
        log.info("Phase A: GRPO on q_φ (SFT model)")
        log.info("=" * 60)
        rc = run_sft_rl(
            config,
            PROJECT_ROOT,
            config_path=args.config.resolve(),
            wandb_enabled=wandb_enabled,
            extra_overrides=verl_overrides,
        )
        if rc != 0:
            return rc

    if args.phase in ("c2f", "both"):
        log.info("=" * 60)
        log.info("Phase B: Supervised fine-tuning of p_θ (C2F model)")
        log.info("=" * 60)
        rc = run_c2f_finetune(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
        if rc != 0:
            return rc

    if args.phase == "joint":
        log.info("=" * 60)
        log.info("Joint: Simultaneous SFT + C2F training")
        log.info("=" * 60)
        rc = run_joint(
            config,
            PROJECT_ROOT,
            config_path=args.config.resolve(),
            wandb_enabled=wandb_enabled,
            extra_overrides=verl_overrides,
        )
        if rc != 0:
            return rc

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
