"""
Build veRL SFT config overrides from experiment YAML and launch args.

Used by scripts/04_sft_train.py to build Hydra overrides and invoke the veRL SFT trainer.
"""
from pathlib import Path
from typing import Any


def build_verl_sft_overrides(
    sft_config: dict[str, Any],
    project_root: Path,
    *,
    wandb_enabled: bool = False,
) -> list[str]:
    """
    Build Hydra override list for veRL SFT from experiment sft config.

    Args:
        sft_config: The 'sft' section from experiment YAML (model, num_gpus, dataset_dir, etc.)
        project_root: Project root for resolving relative paths
        wandb_enabled: Whether W&B logging is enabled (WANDB_* env vars are already set
            by ``setup_wandb``; this flag tells veRL to use the "wandb" logger).

    Returns:
        List of "key=value" strings for Hydra overrides
    """
    import os

    num_gpus = int(sft_config.get("num_gpus", 1))
    model = sft_config.get("model", "Qwen/Qwen3-4B")
    dataset_dir = Path(sft_config.get("dataset_dir", "data/sft_dataset"))
    checkpoint_dir = Path(sft_config.get("checkpoint_dir", "checkpoints/sft"))

    if not dataset_dir.is_absolute():
        dataset_dir = project_root / dataset_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir

    train_parquet = dataset_dir / "train.parquet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"model.partial_pretrain={model}",
        f"data.train_files={train_parquet}",
        f"data.prompt_key=prompt",
        f"data.max_prompt_length={sft_config.get('max_prompt_length', 512)}",
        f"data.max_response_length={sft_config.get('max_response_length', 512)}",
        f"data.train_batch_size={sft_config.get('train_batch_size', 32)}",
        f"optim.lr={sft_config.get('lr', 1e-5)}",
    ]

    total_epochs = sft_config.get("epochs")
    if total_epochs is not None:
        overrides.append(f"trainer.total_epochs={total_epochs}")

    # W&B integration — veRL reads WANDB_* env vars automatically, but we also
    # pass the logger override so veRL's Trainer picks it up via Hydra.
    if wandb_enabled:
        overrides.append("trainer.logger=['wandb']")
        project = os.environ.get("WANDB_PROJECT", "coarse-to-fine")
        overrides.append(f"trainer.project_name={project}")

    return overrides


def get_verl_sft_entrypoint() -> str:
    """
    Return the module path for veRL FSDP SFT trainer (invoked as python -m <path>).

    veRL SFT runs in SPMD mode with torchrun; the entrypoint may be
    verl.trainer.main_fsdp_sft or similar depending on version.
    """
    return "verl.trainer.main_fsdp_sft"
