"""
Build veRL GRPO config overrides from experiment YAML and launch args.

Used by scripts/07_rl_train.py to build Hydra overrides and invoke the veRL
PPO/GRPO trainer for Phase A (GRPO on q_φ, SFT model).

Pattern mirrors src/sft/train.py:build_verl_sft_overrides.
"""
from pathlib import Path
from typing import Any


def build_verl_grpo_overrides(
    rl_sft_config: dict[str, Any],
    project_root: Path,
    *,
    wandb_enabled: bool = False,
) -> list[str]:
    """
    Build Hydra override list for veRL GRPO from the ``rl.sft_rl`` config section.

    Key GRPO-specific settings vs. SFT:
      - ``algorithm.adv_estimator=grpo``
      - ``actor_rollout_ref.rollout.n``    — group size (rollout_n samples / prompt)
      - ``actor_rollout_ref.actor.use_kl_loss=true``
      - ``actor_rollout_ref.actor.kl_loss_coef`` — KL penalty coefficient
      - ``algorithm.use_kl_in_reward=false``  — KL lives in actor loss, not reward

    The custom reward manager (C2FRewardManager) is referenced by path/name so
    veRL can import and instantiate it.  The experiment config path is passed via
    the ``C2F_CONFIG_PATH`` environment variable set in the launch script.

    Args:
        rl_sft_config: The ``rl.sft_rl`` section from the experiment YAML.
        project_root: Project root for resolving relative paths.
        wandb_enabled: Whether W&B logging is enabled.

    Returns:
        List of ``key=value`` strings for Hydra overrides.
    """
    import os

    num_gpus = int(rl_sft_config.get("num_gpus", 1))
    model_path = rl_sft_config.get("model_path", "Qwen/Qwen3-4B")
    dataset_dir = Path(rl_sft_config.get("dataset_dir", "data/rl_dataset"))
    checkpoint_dir = Path(rl_sft_config.get("checkpoint_dir", "checkpoints/rl/sft"))

    if not dataset_dir.is_absolute():
        dataset_dir = project_root / dataset_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir

    train_parquet = dataset_dir / "sft_rl.parquet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        # ── Trainer ─────────────────────────────────────────────────────────
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.default_local_dir={checkpoint_dir}",
        # ── Model ────────────────────────────────────────────────────────────
        f"model.partial_pretrain={model_path}",
        f"model.fsdp_config.model_dtype=bf16",
        # ── Data ─────────────────────────────────────────────────────────────
        f"data.train_files={train_parquet}",
        f"data.val_files={train_parquet}",
        f"data.prompt_key=prompt",
        f"data.max_prompt_length={rl_sft_config.get('max_prompt_length', 64)}",
        f"data.max_response_length={rl_sft_config.get('max_response_length', 256)}",
        f"data.train_batch_size={rl_sft_config.get('train_batch_size', 64)}",
        # ── Algorithm: GRPO ──────────────────────────────────────────────────
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        # ── Actor / Rollout ───────────────────────────────────────────────────
        f"actor_rollout_ref.rollout.n={rl_sft_config.get('rollout_n', 8)}",
        f"actor_rollout_ref.rollout.temperature={rl_sft_config.get('temperature', 1.0)}",
        "actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={rl_sft_config.get('kl_coef', 0.01)}",
        f"actor_rollout_ref.actor.optim.lr={rl_sft_config.get('lr', 1e-6)}",
        # ── Custom reward function ────────────────────────────────────────────
        "reward_model.reward_manager=custom",
        "reward_model.custom_reward_function.path=src/rl/reward.py",
        "reward_model.custom_reward_function.name=C2FRewardManager",
    ]

    total_epochs = rl_sft_config.get("epochs")
    if total_epochs is not None:
        overrides.append(f"trainer.total_epochs={total_epochs}")

    # W&B integration (mirrors build_verl_sft_overrides pattern)
    if wandb_enabled:
        overrides.append("trainer.logger=['console','wandb']")
        project = os.environ.get("WANDB_PROJECT", "coarse-to-fine")
        overrides.append(f"trainer.project_name={project}")
    else:
        overrides.append("trainer.logger=['console']")

    return overrides


def get_verl_grpo_entrypoint() -> str:
    """
    Return the veRL PPO/GRPO trainer module path.

    veRL uses the same ``main_ppo`` entrypoint for both PPO and GRPO; the
    algorithm is selected via ``algorithm.adv_estimator=grpo``.
    """
    return "verl.trainer.main_ppo"
