#!/usr/bin/env python3
"""
Step 7: ELBO Optimisation — GRPO on q_φ / Supervised on p_θ.

Alternates between two training phases to optimise the ELBO:

  Phase A (``--phase sft``):
    GRPO on q_φ (SFT model), p_θ (C2F) frozen.
    Reward = log p_θ(x, z) (from C2F forward pass) + format bonus.
    KL(q_φ ‖ q_φ_ref) is handled by veRL actor.use_kl_loss=true.

  Phase B (``--phase c2f``):
    Supervised fine-tuning of p_θ (C2F), q_φ (SFT) frozen.
    Directly maximises E_{q_φ}[log p_θ(x, z)] — no custom GRPO loop needed.

  Both (``--phase both``):
    Runs Phase A then Phase B sequentially (one round of alternation).

Usage:
    # One round, one phase at a time:
    python scripts/07_rl_train.py --phase sft  --config config/experiments/latent_generation.yaml
    python scripts/07_rl_train.py --phase c2f  --config config/experiments/latent_generation.yaml

    # One full round (Phase A → Phase B):
    python scripts/07_rl_train.py --phase both --config config/experiments/latent_generation.yaml

    # Smoke-test Phase A (small batch, few epochs):
    python scripts/07_rl_train.py --phase sft \\
        --config config/experiments/latent_generation.yaml \\
        rl.sft_rl.epochs=1 rl.sft_rl.train_batch_size=8 rl.sft_rl.rollout_n=4

    # Smoke-test Phase B (100 samples, 1 epoch):
    python scripts/07_rl_train.py --phase c2f \\
        --config config/experiments/latent_generation.yaml \\
        rl.c2f_finetune.num_samples=100 rl.c2f_finetune.epochs=1

Config overrides:
    ``rl.sft_rl.*`` and ``rl.c2f_finetune.*`` dot-path overrides are applied to
    the in-memory config before Phase A / Phase B are launched.  Any override
    that does not start with ``rl.`` is passed through as a Hydra override
    directly to the veRL trainer (Phase A only).

Requires: pip install -e ".[rl]" and a veRL-compatible environment.
Dataset:  data/sft_dataset/train.parquet (from step 3) must exist.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _cast_value(raw: str):
    """Cast a string override value to int / float / bool / None / str."""
    if raw.lower() == "null":
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_overrides(
    config: dict, overrides: list[str]
) -> tuple[dict, list[str]]:
    """
    Split overrides into config-dict updates (``rl.*``) and veRL pass-throughs.

    ``rl.sft_rl.epochs=1`` updates ``config['rl']['sft_rl']['epochs'] = 1``.
    Everything else is collected in ``verl_overrides`` and passed to torchrun.

    Returns:
        (updated_config, verl_overrides)
    """
    verl_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            verl_overrides.append(override)
            continue
        key_path, _, raw_value = override.partition("=")
        parts = key_path.split(".")
        if parts[0] != "rl":
            verl_overrides.append(override)
            continue
        # Navigate / create nested dict nodes
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _cast_value(raw_value)
    return config, verl_overrides


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 7: ELBO optimisation — GRPO on q_φ and/or supervised fine-tuning on p_θ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["sft", "c2f", "both"],
        required=True,
        help="Phase to run: sft (Phase A / GRPO), c2f (Phase B / SFT), or both",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiments" / "latent_generation.yaml",
        help="Path to experiment YAML with 'rl' section",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="OVERRIDE",
        help=(
            "Config overrides: rl.sft_rl.epochs=1 (applied to YAML config) "
            "or Hydra-style overrides for veRL (Phase A only)"
        ),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config not found: {args.config}", file=sys.stderr)
        return 1

    # ── Environment and W&B ─────────────────────────────────────────────────
    from src.utils.env import load_env, setup_wandb

    load_env()
    config = _load_config(args.config)
    wandb_enabled = setup_wandb(config, step_name=f"rl-{args.phase}")
    if wandb_enabled:
        print(f"W&B logging enabled for rl-{args.phase}")

    if "rl" not in config:
        print("Error: Config missing 'rl' section", file=sys.stderr)
        return 1

    # ── Apply dot-path overrides ─────────────────────────────────────────────
    config, verl_overrides = _apply_overrides(config, args.overrides)

    # ── Run phases ───────────────────────────────────────────────────────────
    from src.rl.train import run_c2f_finetune, run_sft_rl

    if args.phase in ("sft", "both"):
        print("=" * 60)
        print("Phase A: GRPO on q_φ (SFT model)")
        print("=" * 60)
        rc = run_sft_rl(
            config,
            PROJECT_ROOT,
            wandb_enabled=wandb_enabled,
            extra_overrides=verl_overrides,
        )
        if rc != 0:
            print(f"Phase A failed (return code {rc})", file=sys.stderr)
            return rc

    if args.phase in ("c2f", "both"):
        print("=" * 60)
        print("Phase B: Supervised fine-tuning of p_θ (C2F model)")
        print("=" * 60)
        rc = run_c2f_finetune(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
        if rc != 0:
            print(f"Phase B failed (return code {rc})", file=sys.stderr)
            return rc

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
