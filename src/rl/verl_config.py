"""
Build veRL config overrides from experiment YAML and launch args.

Used by scripts/07_rl_train.py to build Hydra overrides and invoke the veRL
PPO/GRPO/REINFORCE trainer.

Pattern mirrors scripts/04_sft_train.py:build_overrides.
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

    # Resolve model_path to absolute so veRL can load tokenizer/config from it.
    # If it looks like an HF hub id (no path separators, not a local dir) leave as-is.
    model_path_resolved = Path(model_path)
    if not model_path_resolved.is_absolute() and (
        "/" in model_path
        or model_path_resolved.exists()
        or (project_root / model_path_resolved).exists()
    ):
        candidate = project_root / model_path_resolved
        if candidate.exists():
            model_path = str(candidate)

    train_parquet = dataset_dir / "sft_rl.parquet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        # ── Trainer ─────────────────────────────────────────────────────────
        f"++trainer.n_gpus_per_node={num_gpus}",
        f"++trainer.default_local_dir={checkpoint_dir}",
        # ── Model (main_ppo uses actor_rollout_ref.model.*, not model.*) ───────
        f"++actor_rollout_ref.model.path={model_path}",
        "++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        # ── Data ─────────────────────────────────────────────────────────────
        f"++data.train_files={train_parquet}",
        f"++data.val_files={train_parquet}",
        "++data.prompt_key=prompt",
        f"++data.max_prompt_length={rl_sft_config.get('max_prompt_length', 256)}",
        f"++data.max_response_length={rl_sft_config.get('max_response_length', 256)}",
        f"++data.train_batch_size={rl_sft_config.get('train_batch_size', 64)}",
        # Validation batch defaults to len(val_dataset) (ray_trainer.py:366-368),
        # which ships the entire val split in one generate_sequences call. Cap.
        f"++data.val_batch_size={rl_sft_config.get('val_batch_size', 64)}",
        f"++data.dataloader_num_workers={rl_sft_config.get('dataloader_num_workers', 4)}",
        # ── Algorithm: GRPO ──────────────────────────────────────────────────
        "++algorithm.adv_estimator=grpo",
        "++algorithm.use_kl_in_reward=false",
        # ── Actor / Rollout ───────────────────────────────────────────────────
        "++actor_rollout_ref.rollout.name=vllm",
        f"++actor_rollout_ref.rollout.n={rl_sft_config.get('rollout_n', 8)}",
        f"++actor_rollout_ref.rollout.temperature={rl_sft_config.get('temperature', 1.0)}",
        f"++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rl_sft_config.get('ppo_micro_batch_size_per_gpu', 8)}",
        # vLLM rollout KV-cache budget. Qwen3-4B's native max_position_embeddings
        # is 40960 — leaving these unset OOMs a single H100 colocated with FSDP
        # actor + ref + reward models. Prompt+response = 512 by default, so
        # max_model_len=512 is exact; gpu_memory_utilization=0.3 reserves ~42
        # GiB on a 141GiB H100, leaving room for the ~83 GiB the actor/ref/reward
        # already hold at engine init. Tune per-experiment via rl.<section>.*.
        f"++actor_rollout_ref.rollout.max_model_len={rl_sft_config.get('max_prompt_length', 256) + rl_sft_config.get('max_response_length', 256)}",
        f"++actor_rollout_ref.rollout.max_num_seqs={rl_sft_config.get('rollout_max_num_seqs', 64)}",
        f"++actor_rollout_ref.rollout.max_num_batched_tokens={rl_sft_config.get('rollout_max_num_batched_tokens', 4096)}",
        f"++actor_rollout_ref.rollout.gpu_memory_utilization={rl_sft_config.get('rollout_gpu_memory_utilization', 0.3)}",
        "++actor_rollout_ref.actor.use_kl_loss=true",
        f"++actor_rollout_ref.actor.kl_loss_coef={rl_sft_config.get('kl_coef', 0.01)}",
        # Nonzero entropy_coeff enables `actor/entropy` in W&B (ray_trainer.py:1221
        # gates the compute+log on entropy_coeff != 0). 1e-8 is numerically
        # negligible in the loss (dp_actor.py:650: policy_loss -= entropy * coef)
        # but turns the metric on so we can watch q_φ's rollout entropy.
        f"++actor_rollout_ref.actor.entropy_coeff={rl_sft_config.get('entropy_coeff', 1e-8)}",
        f"++actor_rollout_ref.actor.optim.lr={rl_sft_config.get('lr', 1e-6)}",
        f"++actor_rollout_ref.actor.ppo_mini_batch_size={rl_sft_config.get('train_batch_size', 64)}",
        f"++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={rl_sft_config.get('ppo_micro_batch_size_per_gpu', 8)}",
        f"++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={rl_sft_config.get('ppo_micro_batch_size_per_gpu', 8)}",
        # ── Custom reward manager (experimental reward_loop schema) ──────────
        # veRL 0.7 has two parallel systems: legacy trainer/ppo/reward.py reads
        # `reward.reward_manager.*`; the agent_loop path (what rollouts actually
        # hit) reads `reward_model.*` via verl/experimental/reward_loop/reward_loop.py.
        # If these keys aren't set correctly the loader silently falls back to
        # NaiveRewardManager (which expects a different parquet schema and crashes).
        "++reward_model.enable=false",
        "++reward_model.num_workers=1",
        "++reward_model.reward_loop_source=import",
        "++reward_model.reward_loop_class_name=C2FRewardManager",
        f"++reward_model.reward_loop_module_path={project_root / 'src' / 'rl' / 'reward_sft.py'}",
    ]

    total_epochs = rl_sft_config.get("epochs")
    if total_epochs is not None:
        overrides.append(f"++trainer.total_epochs={total_epochs}")

    # W&B integration (mirrors SFT override pattern)
    if wandb_enabled:
        overrides.append("++trainer.logger=['console','wandb']")
        project = os.environ.get("WANDB_PROJECT", "coarse-to-fine")
        overrides.append(f"++trainer.project_name={project}")
        # Propagate the timestamped run name (set by setup_wandb) into veRL,
        # otherwise its W&B logger invents its own (the infamous "gsm8k").
        wandb_run_name = os.environ.get("WANDB_NAME", "")
        if wandb_run_name:
            overrides.append(f"++trainer.experiment_name={wandb_run_name}")
    else:
        overrides.append("++trainer.logger=['console']")

    return overrides


def build_verl_joint_overrides(
    rl_joint_config: dict[str, Any],
    project_root: Path,
    *,
    wandb_enabled: bool = False,
) -> list[str]:
    """
    Build Hydra override list for veRL REINFORCE from ``rl.joint`` config.

    Key differences from :func:`build_verl_grpo_overrides`:
      - ``algorithm.adv_estimator=reinforce_plus_plus`` (not GRPO)
      - ``actor_rollout_ref.rollout.n=1`` (single sample per prompt)
      - No KL loss — simplest REINFORCE for posterior collapse experiment
      - Points to ``JointC2FRewardManager`` (trains p alongside q)
    """
    import os

    num_gpus = int(rl_joint_config.get("num_gpus", 1))
    model_path = rl_joint_config.get("model_path", "Qwen/Qwen3-4B")
    dataset_dir = Path(rl_joint_config.get("dataset_dir", "data/rl_dataset"))
    checkpoint_dir = Path(rl_joint_config.get("checkpoint_dir", "checkpoints/rl/joint"))

    if not dataset_dir.is_absolute():
        dataset_dir = project_root / dataset_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir

    model_path_resolved = Path(model_path)
    if not model_path_resolved.is_absolute() and (
        "/" in model_path
        or model_path_resolved.exists()
        or (project_root / model_path_resolved).exists()
    ):
        candidate = project_root / model_path_resolved
        if candidate.exists():
            model_path = str(candidate)

    train_parquet = dataset_dir / "sft_rl.parquet"
    val_parquet = dataset_dir / "sft_rl_val.parquet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        # ── Trainer ─────────────────────────────────────────────────────────
        f"++trainer.n_gpus_per_node={num_gpus}",
        f"++trainer.default_local_dir={checkpoint_dir}",
        # ── Model ───────────────────────────────────────────────────────────
        f"++actor_rollout_ref.model.path={model_path}",
        "++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        # ── Data ────────────────────────────────────────────────────────────
        f"++data.train_files={train_parquet}",
        f"++data.val_files={val_parquet}",
        "++data.prompt_key=prompt",
        f"++data.max_prompt_length={rl_joint_config.get('max_prompt_length', 256)}",
        f"++data.max_response_length={rl_joint_config.get('max_response_length', 256)}",
        f"++data.train_batch_size={rl_joint_config.get('train_batch_size', 256)}",
        # Validation batch defaults to len(val_dataset) (ray_trainer.py:366-368),
        # which ships the entire val split in one generate_sequences call. Cap.
        f"++data.val_batch_size={rl_joint_config.get('val_batch_size', 64)}",
        f"++data.dataloader_num_workers={rl_joint_config.get('dataloader_num_workers', 4)}",
        # ── Algorithm: REINFORCE++ (no critic, no KL) ───────────────────────
        "++algorithm.adv_estimator=reinforce_plus_plus",
        "++algorithm.use_kl_in_reward=false",
        # ── Actor / Rollout ─────────────────────────────────────────────────
        "++actor_rollout_ref.rollout.name=vllm",
        "++actor_rollout_ref.rollout.n=1",
        f"++actor_rollout_ref.rollout.tensor_model_parallel_size={num_gpus}",
        "++actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
        f"++actor_rollout_ref.rollout.temperature={rl_joint_config.get('temperature', 1.0)}",
        f"++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rl_joint_config.get('ppo_micro_batch_size_per_gpu', 16)}",
        # vLLM rollout KV-cache budget. Qwen3-4B's native max_position_embeddings
        # is 40960 — leaving these unset OOMs a single H100 colocated with FSDP
        # actor + ref + JointC2FRewardManager p_θ. Prompt+response = 512 by
        # default, so max_model_len=512 is exact; gpu_memory_utilization=0.3
        # reserves ~42 GiB on a 141GiB H100, leaving room for the ~83 GiB the
        # actor/ref/reward already hold at engine init. Tune via rl.joint.*.
        f"++actor_rollout_ref.rollout.max_model_len={rl_joint_config.get('max_prompt_length', 256) + rl_joint_config.get('max_response_length', 256)}",
        f"++actor_rollout_ref.rollout.max_num_seqs={rl_joint_config.get('rollout_max_num_seqs', 64)}",
        f"++actor_rollout_ref.rollout.max_num_batched_tokens={rl_joint_config.get('rollout_max_num_batched_tokens', 4096)}",
        f"++actor_rollout_ref.rollout.gpu_memory_utilization={rl_joint_config.get('rollout_gpu_memory_utilization', 0.3)}",
        "++actor_rollout_ref.actor.use_kl_loss=false",
        # Enable `actor/entropy` logging so posterior-collapse demonstrations can
        # show q_φ entropy decaying alongside p_loss going down. 1e-8 is
        # numerically ≈ 0 in the loss, so the collapse dynamics are preserved.
        f"++actor_rollout_ref.actor.entropy_coeff={rl_joint_config.get('entropy_coeff', 1e-8)}",
        f"++actor_rollout_ref.actor.optim.lr={rl_joint_config.get('lr', 1e-6)}",
        f"++actor_rollout_ref.actor.ppo_mini_batch_size={rl_joint_config.get('train_batch_size', 256)}",
        f"++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={rl_joint_config.get('ppo_micro_batch_size_per_gpu', 16)}",
        f"++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={rl_joint_config.get('ppo_micro_batch_size_per_gpu', 16)}",
        # ── Reward manager (experimental reward_loop schema) ─────────────────
        # veRL 0.7 has two parallel systems: legacy trainer/ppo/reward.py reads
        # `reward.reward_manager.*`; the agent_loop path (what rollouts actually
        # hit) reads `reward_model.*` via verl/experimental/reward_loop/reward_loop.py.
        # If these keys aren't set correctly the loader silently falls back to
        # NaiveRewardManager (which expects a different parquet schema and crashes).
        #
        # num_workers=1 so a single p_θ is shared across all rollouts — required
        # for correct joint ELBO updates (otherwise each RewardLoopWorker trains
        # an independent copy of p_θ and q_φ's REINFORCE sees biased rewards).
        "++reward_model.enable=false",
        "++reward_model.num_workers=1",
        "++reward_model.reward_loop_source=import",
        "++reward_model.reward_loop_class_name=JointC2FRewardManager",
        f"++reward_model.reward_loop_module_path={project_root / 'src' / 'rl' / 'reward_joint.py'}",
    ]

    total_epochs = rl_joint_config.get("epochs")
    if total_epochs is not None:
        overrides.append(f"++trainer.total_epochs={total_epochs}")

    if wandb_enabled:
        overrides.append("++trainer.logger=['console','wandb']")
        project = os.environ.get("WANDB_PROJECT", "coarse-to-fine")
        overrides.append(f"++trainer.project_name={project}")
        # Propagate the timestamped run name (set by setup_wandb) into veRL,
        # otherwise its W&B logger invents its own (the infamous "gsm8k").
        wandb_run_name = os.environ.get("WANDB_NAME", "")
        if wandb_run_name:
            overrides.append(f"++trainer.experiment_name={wandb_run_name}")
    else:
        overrides.append("++trainer.logger=['console']")

    return overrides
