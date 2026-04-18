"""
Environment and secrets loading utilities.

Loads .env from the project root and configures W&B from the experiment config.
All scripts should call ``load_env()`` early, then ``setup_wandb(config)`` before
any training to ensure W&B is properly initialized.
"""

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env(env_path: Path | None = None) -> None:
    """
    Load environment variables from a .env file.

    Falls back to PROJECT_ROOT/.env if no path is given.
    Silently skips if the file does not exist (e.g., on HPC where secrets
    are injected via the environment or a module system).

    Uses a simple key=value parser — no third-party dependency needed.
    """
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip inline comments (e.g. "value  # comment") so they aren't passed to Hydra etc.
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            # Don't overwrite existing env vars (system takes precedence)
            if key not in os.environ:
                os.environ[key] = value


def setup_wandb(config: dict[str, Any], step_name: str | None = None) -> bool:
    """
    Configure Weights & Biases from the experiment config's ``wandb:`` section.

    Sets WANDB_* environment variables so that both HF Trainer and veRL pick up
    the settings automatically.  Returns True if W&B is enabled, False otherwise.

    Expected config layout::

        wandb:
          enabled: true
          project: "coarse-to-fine"
          entity: null          # uses WANDB_ENTITY env var or default
          group: "latent-gen"   # optional grouping of related runs
          tags: ["sft", "qwen3-4b"]

    Individual steps can override ``run_name`` in their own section
    (``sft.run_name``, ``c2f_training.run_name``).

    Args:
        config: Full experiment config dict.
        step_name: Optional step identifier (e.g. "sft", "c2f-pretrain").
            Appended to run_name for clarity when multiple steps log to the
            same W&B project.

    Returns:
        True if W&B logging is enabled, False if disabled.
    """
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        os.environ.setdefault("WANDB_DISABLED", "true")
        return False

    # WandB requires API keys to be 40+ characters; invalid/placeholder keys cause a crash later
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key or len(api_key) < 40:
        from src.common.logging import get_logger

        get_logger(__name__).warning(
            "W&B disabled: WANDB_API_KEY is missing or too short (WandB requires 40+ chars). "
            "Get a key from https://wandb.ai/authorize and set it in .env."
        )
        os.environ.setdefault("WANDB_DISABLED", "true")
        return False

    # Unset the disabled flag if it was previously set
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = wandb_config.get("mode", "online")

    # Project: use config if set, else .env WANDB_PROJECT, else default
    project = wandb_config.get("project")
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "diffusion-rl")
    os.environ["WANDB_PROJECT"] = project

    # Entity: use config if set; otherwise .env WANDB_ENTITY (from load_env) is used
    entity = wandb_config.get("entity")
    if entity:
        os.environ["WANDB_ENTITY"] = entity

    group = wandb_config.get("group")
    if group:
        os.environ["WANDB_RUN_GROUP"] = group

    tags = wandb_config.get("tags", [])
    if step_name:
        tags = [*tags, step_name]
    if tags:
        os.environ["WANDB_TAGS"] = ",".join(str(t) for t in tags)

    # Log the full experiment config as W&B config metadata
    os.environ.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))

    return True
