"""
ELBO optimisation training functions for Step 7.

Phase A — ``run_sft_rl()``:
    GRPO on q_φ (SFT model), p_θ (C2F) frozen.

Phase B — ``run_c2f_finetune()``:
    Supervised fine-tuning of p_θ (C2F), q_φ (SFT) frozen.

Phase Joint — ``run_joint()``:
    Simultaneous SFT + C2F training (placeholder for custom veRL modification).
"""
import sys
import subprocess
from pathlib import Path
from typing import Any


# ── Override utilities ───────────────────────────────────────────────────────


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


def apply_overrides(
    config: dict, overrides: list[str]
) -> tuple[dict, list[str]]:
    """
    Split overrides into config-dict updates (``rl.*``) and veRL pass-throughs.

    ``rl.sft_rl.epochs=1`` updates ``config['rl']['sft_rl']['epochs'] = 1``.
    Everything else is collected as veRL Hydra overrides.

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
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _cast_value(raw_value)
    return config, verl_overrides


# ── Helpers ──────────────────────────────────────────────────────────────────


def _prep_rl_parquet(config: dict[str, Any], project_root: Path, rl_section: str = "sft_rl") -> Path:
    """
    Build the RL training parquet from the RL split JSONL.

    Reads raw documents from ``dataset.rl_split`` and creates a veRL GRPO
    parquet with columns:
      - ``prompt``: the raw document text (input to q_φ during rollout).
      - ``ground_truth``: copy of ``prompt`` (used by the reward manager
        to run the C2F forward pass).
      - ``data_source``: literal ``"latent_generation"``.

    Returns:
        Path to the saved RL parquet (skips regeneration if it already exists).
    """
    from datasets import Dataset

    rl_sft_cfg = config["rl"][rl_section]

    rl_dataset_dir = Path(rl_sft_cfg.get("dataset_dir", "data/rl_dataset"))
    if not rl_dataset_dir.is_absolute():
        rl_dataset_dir = project_root / rl_dataset_dir
    rl_dataset_dir.mkdir(parents=True, exist_ok=True)

    rl_parquet = rl_dataset_dir / "sft_rl.parquet"

    if rl_parquet.exists():
        print(f"  RL parquet already exists: {rl_parquet}")
        return rl_parquet

    # Load raw documents from the RL split
    dataset_cfg = config.get("dataset", {})
    data_dir = Path(dataset_cfg.get("data_dir", "data/tinystoriesv2_shuffled"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    rl_split_file = data_dir / dataset_cfg.get("rl_split", "tinystoriesv2.rl.jsonl")

    if not rl_split_file.exists():
        raise FileNotFoundError(
            f"RL split not found: {rl_split_file}\n"
            "Run step 0 (00_prepare_data.py) first."
        )

    from src.generation.dataset import load_documents_from_jsonl

    print(f"  Preparing RL parquet from {rl_split_file}...")
    docs = load_documents_from_jsonl([rl_split_file])

    records = [
        {"prompt": doc, "ground_truth": doc, "data_source": "latent_generation"}
        for doc in docs
    ]
    ds = Dataset.from_list(records)
    ds.to_parquet(str(rl_parquet))
    print(f"  Saved RL parquet: {rl_parquet} ({len(ds):,} samples)")
    return rl_parquet


# ── Phase A ───────────────────────────────────────────────────────────────────


def run_sft_rl(
    config: dict[str, Any],
    project_root: Path,
    *,
    config_path: str | Path | None = None,
    wandb_enabled: bool = False,
    extra_overrides: list[str] | None = None,
) -> int:
    """
    Phase A: GRPO on q_φ (SFT model) with p_θ (C2F) frozen.

    Steps:
      1. Validate that required checkpoints exist.
      2. Prepare the RL parquet (add ground_truth / data_source columns).
      3. Build GRPO Hydra overrides via :func:`build_verl_grpo_overrides`.
      4. Launch: ``torchrun --nproc_per_node=N -m verl.trainer.main_ppo <overrides>``

    The ``C2F_CONFIG_PATH`` environment variable is set before launching so
    ``C2FRewardManager.__init__`` can locate the experiment YAML.

    Args:
        config: Full experiment config.
        project_root: Project root directory.
        wandb_enabled: Whether W&B is enabled.
        extra_overrides: Additional Hydra overrides from the CLI.

    Returns:
        Process returncode (0 on success).
    """
    import os

    from src.rl.verl_config import build_verl_grpo_overrides

    rl_sft_cfg = config.get("rl", {}).get("sft_rl", {})
    if not rl_sft_cfg:
        print("Error: config['rl']['sft_rl'] section is missing.", file=sys.stderr)
        return 1

    num_gpus = int(rl_sft_cfg.get("num_gpus", 1))

    # ── Validate checkpoint paths ────────────────────────────────────────────
    c2f_model_path = Path(rl_sft_cfg.get("c2f_model_path", "checkpoints/decoder"))
    if not c2f_model_path.is_absolute():
        c2f_model_path = project_root / c2f_model_path
    if not c2f_model_path.exists():
        print(f"Error: C2F model not found: {c2f_model_path}", file=sys.stderr)
        print(
            "Run step 6 (06_train_decoder.py) first to pretrain the C2F model.",
            file=sys.stderr,
        )
        return 1

    sft_model_path = Path(rl_sft_cfg.get("model_path", "checkpoints/sft"))
    if not sft_model_path.is_absolute():
        sft_model_path = project_root / sft_model_path
    if not sft_model_path.exists():
        print(f"Error: SFT model not found: {sft_model_path}", file=sys.stderr)
        print("Run step 4 (04_sft_train.py) first.", file=sys.stderr)
        return 1

    # ── Prepare RL dataset ───────────────────────────────────────────────────
    print("Phase A: Preparing RL dataset...")
    _prep_rl_parquet(config, project_root)

    # ── Build and launch GRPO ────────────────────────────────────────────────
    overrides = build_verl_grpo_overrides(
        rl_sft_cfg, project_root, wandb_enabled=wandb_enabled
    )
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "verl.trainer.main_ppo",
        *overrides,
    ]
    print("Phase A — GRPO on q_φ:")
    print("  Command:", " ".join(cmd))

    # Export config path so the reward manager can locate the experiment YAML
    resolved_config_path = str(config_path) if config_path else str(
        project_root / "config" / "latent_generation.yaml"
    )
    env = {**os.environ, "C2F_CONFIG_PATH": resolved_config_path}
    return subprocess.run(cmd, cwd=project_root, env=env).returncode


# ── Phase B ───────────────────────────────────────────────────────────────────


def run_c2f_finetune(
    config: dict[str, Any],
    project_root: Path,
    *,
    wandb_enabled: bool = False,
) -> int:
    """
    Phase B: Supervised gradient on p_θ (C2F) using samples from frozen q_φ.

    Implements ``∇_θ L = E_{q_φ}[∇_θ log p_θ(x, z)]``.

    Steps:
      1. Load prompts from the SFT training parquet.
      2. Generate z ~ q_φ(·|x) with the frozen RL-updated SFT checkpoint.
         (Uses :func:`src.generation.inference.generate`.)
      3. Verify and filter outputs; flatten for C2F training.
         (Reuses :func:`src.generation.dataset.verify_and_filter_outputs` and
         :func:`src.generation.dataset.flatten_for_c2f`.)
      4. Fine-tune C2F on the freshly generated parquet.
         (Reuses :class:`src.c2f_training.train.C2FTrainer`.)

    Args:
        config: Full experiment config.
        project_root: Project root directory.
        wandb_enabled: Whether W&B is enabled.

    Returns:
        Process returncode (0 on success).
    """
    from datasets import load_dataset

    from src.c2f_training.dataset import C2FDataset
    from src.c2f_training.tokenizer import load_or_train_space_tokenizer
    from src.c2f_training.train import C2FTrainer, build_training_args, load_c2f_model
    from src.generation.dataset import flatten_for_c2f, verify_and_filter_outputs
    from src.generation.inference import generate

    c2f_ft_cfg = config.get("rl", {}).get("c2f_finetune", {})
    if not c2f_ft_cfg:
        print("Error: config['rl']['c2f_finetune'] section is missing.", file=sys.stderr)
        return 1

    # ── Resolve paths ────────────────────────────────────────────────────────
    def _abs(rel: str, default: str) -> Path:
        p = Path(c2f_ft_cfg.get(rel, default))
        return p if p.is_absolute() else project_root / p

    sft_model_path = _abs("sft_model_path", "checkpoints/rl/sft")
    c2f_model_path = _abs("model_path", "checkpoints/decoder")
    generation_output_dir = _abs("generation_output_dir", "data/rl_dataset/c2f_finetune")
    checkpoint_dir = _abs("checkpoint_dir", "checkpoints/rl/c2f")

    sft_dataset_dir = Path(config.get("sft", {}).get("dataset_dir", "data/sft_dataset"))
    if not sft_dataset_dir.is_absolute():
        sft_dataset_dir = project_root / sft_dataset_dir

    if not sft_model_path.exists():
        print(f"Error: SFT model not found: {sft_model_path}", file=sys.stderr)
        print("Run Phase A (--phase sft) first.", file=sys.stderr)
        return 1

    # ── Step 1: Load prompts ─────────────────────────────────────────────────
    sft_parquet = sft_dataset_dir / "train.parquet"
    if not sft_parquet.exists():
        print(f"Error: SFT parquet not found: {sft_parquet}", file=sys.stderr)
        return 1

    ds = load_dataset("parquet", data_files=str(sft_parquet), split="train")
    prompts: list[str] = list(ds["prompt"])
    num_samples = c2f_ft_cfg.get("num_samples")
    if num_samples is not None:
        prompts = prompts[:int(num_samples)]
    print(f"Phase B — Step 1: Generating z ~ q_φ for {len(prompts):,} prompts...")

    # ── Step 2: Generate z ~ q_φ(·|x) with frozen SFT ──────────────────────
    gen_config = config.get("generation", {})
    num_gpus = int(c2f_ft_cfg.get("num_gpus", gen_config.get("num_gpus", 1)))

    outputs: list[str] = generate(
        backend="vllm",
        model_path=str(sft_model_path),
        prompts=prompts,
        num_gpus=num_gpus,
        max_tokens=gen_config.get("max_tokens", 256),
        temperature=gen_config.get("temperature", 0.7),
        top_p=gen_config.get("top_p", 0.9),
        top_k=gen_config.get("top_k", -1),
        repetition_penalty=gen_config.get("repetition_penalty", 1.0),
        seed=gen_config.get("seed", 42),
    )

    # ── Step 3: Verify and flatten ───────────────────────────────────────────
    print("Phase B — Step 2: Verifying and flattening outputs...")
    filtered_prompts, filtered_outputs, stats = verify_and_filter_outputs(
        prompts, outputs, config
    )
    print(str(stats))

    if not filtered_prompts:
        print("Error: No valid outputs after verification.", file=sys.stderr)
        return 1

    c2f_parquet_path = flatten_for_c2f(
        filtered_prompts,
        filtered_outputs,
        output_dir=generation_output_dir,
        filename="c2f_finetune.parquet",
    )
    print(f"  Saved {len(filtered_prompts):,} samples to {c2f_parquet_path}")

    # ── Step 4: Fine-tune C2F ────────────────────────────────────────────────
    print("Phase B — Step 3: Fine-tuning C2F model on q_φ samples...")

    c2f_train_cfg = config.get("c2f_training", {})
    tokenizer_dir = Path(
        c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
    )
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = project_root / tokenizer_dir

    space_tokenizer = load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=generation_output_dir,
        dataset_format="c2f",
        parquet_filename="c2f_finetune.parquet",
    )
    vocab_size = space_tokenizer.vocab_size

    # Build a fine-tuning config that points at the new parquet and uses the
    # rl.c2f_finetune overrides (lr, epochs, batch size, fsdp, checkpoint_dir).
    ft_c2f_cfg: dict[str, Any] = {
        **c2f_train_cfg,
        "init_from": str(c2f_model_path),
        "dataset_dir": str(generation_output_dir),
        "dataset_format": "c2f",
        "parquet_filename": "c2f_finetune.parquet",
        "checkpoint_dir": str(checkpoint_dir),
        "per_device_batch_size": c2f_ft_cfg.get(
            "per_device_batch_size", c2f_train_cfg.get("per_device_batch_size", 8)
        ),
        "gradient_accumulation_steps": c2f_ft_cfg.get(
            "gradient_accumulation_steps",
            c2f_train_cfg.get("gradient_accumulation_steps", 4),
        ),
        "lr": c2f_ft_cfg.get("lr", c2f_train_cfg.get("lr", 5e-5)),
        "epochs": c2f_ft_cfg.get("epochs", 3),
        "fsdp": c2f_ft_cfg.get("fsdp", c2f_train_cfg.get("fsdp", "full_shard")),
    }
    ft_config = {**config, "c2f_training": ft_c2f_cfg}

    model = load_c2f_model(ft_config, vocab_size=vocab_size)

    full_dataset = C2FDataset(
        data_dir=str(generation_output_dir),
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config["word_count_constraints"],
        text_word_count=config.get("text_word_count", 32),
        parquet_filename="c2f_finetune.parquet",
        dataset_format="c2f",
        tokenizer=space_tokenizer,
    )

    splits = full_dataset.train_test_split(
        test_size=c2f_train_cfg.get("eval_split", 0.05),
        seed=c2f_train_cfg.get("seed", 42),
    )

    training_args = build_training_args(ft_config, project_root, wandb_enabled=wandb_enabled)

    trainer = C2FTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        scale_lengths=config["scale_lengths"],
    )

    trainer.train()
    trainer.save_model()
    print(f"Phase B complete. C2F model saved to: {training_args.output_dir}")
    return 0


# ── Phase Joint ──────────────────────────────────────────────────────────────


def run_joint(
    config: dict[str, Any],
    project_root: Path,
    *,
    config_path: str | Path | None = None,
    wandb_enabled: bool = False,
    extra_overrides: list[str] | None = None,
) -> int:
    """
    Joint ELBO training: REINFORCE on q_φ + supervised MLE on p_θ simultaneously.

    p_θ is trained inside the ``JointC2FRewardManager``: on each batch the
    reward manager computes ``reward = log p_θ(x, z) / num_tokens`` for veRL's
    REINFORCE update on q_φ, then does a backward pass and optimizer step on p_θ.

    Expected outcome (posterior collapse sanity check): q_φ collapses to
    degenerate latents while p_θ learns to ignore z and becomes a plain LM
    over x.

    Args:
        config: Full experiment config.
        project_root: Project root directory.
        config_path: Path to experiment YAML (for reward manager).
        wandb_enabled: Whether W&B is enabled.
        extra_overrides: Additional Hydra overrides from the CLI.

    Returns:
        Process returncode (0 on success).
    """
    import os

    from src.rl.verl_config import build_verl_joint_overrides

    rl_joint_cfg = config.get("rl", {}).get("joint", {})
    if not rl_joint_cfg:
        print("Error: config['rl']['joint'] section is missing.", file=sys.stderr)
        return 1

    num_gpus = int(rl_joint_cfg.get("num_gpus", 1))

    # ── Validate checkpoint paths ────────────────────────────────────────────
    c2f_model_path = Path(rl_joint_cfg.get("c2f_model_path", "checkpoints/decoder"))
    if not c2f_model_path.is_absolute():
        c2f_model_path = project_root / c2f_model_path
    if not c2f_model_path.exists():
        print(f"Error: C2F model not found: {c2f_model_path}", file=sys.stderr)
        print("Run step 6 (06_train_decoder.py) first.", file=sys.stderr)
        return 1

    sft_model_path = Path(rl_joint_cfg.get("model_path", "checkpoints/sft"))
    if not sft_model_path.is_absolute():
        sft_model_path = project_root / sft_model_path
    if not sft_model_path.exists():
        print(f"Error: SFT model not found: {sft_model_path}", file=sys.stderr)
        print("Run step 4 (04_sft_train.py) first.", file=sys.stderr)
        return 1

    # ── Prepare RL dataset ───────────────────────────────────────────────────
    print("Joint: Preparing RL dataset...")
    _prep_rl_parquet(config, project_root, rl_section="joint")

    # ── Build and launch REINFORCE ───────────────────────────────────────────
    overrides = build_verl_joint_overrides(
        rl_joint_cfg, project_root, wandb_enabled=wandb_enabled,
    )
    if extra_overrides:
        overrides.extend(extra_overrides)

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "-m", "verl.trainer.main_ppo",
            *overrides,
        ]
    else:
        # Single GPU: use plain python to avoid torchrun setting
        # MASTER_ADDR/MASTER_PORT env vars that confuse Ray workers.
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            *overrides,
        ]
    print("Joint — REINFORCE on q_φ + MLE on p_θ:")
    print("  Command:", " ".join(cmd))

    resolved_config_path = str(config_path) if config_path else str(
        project_root / "config" / "latent_generation.yaml"
    )
    env = {
        **os.environ,
        "C2F_CONFIG_PATH": resolved_config_path,
        # Prevent Ray from setting CUDA_VISIBLE_DEVICES="" on processes with
        # num_gpus=0.  The reward manager needs GPU access for p_θ training.
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    }
    if num_gpus == 1:
        # Single-GPU: FSDP broadcasts are no-ops but still init NCCL.
        # Use socket transport over loopback to avoid HPC-specific issues.
        env.update({
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_NET": "Socket",
            "NCCL_SOCKET_IFNAME": "lo",
        })
    return subprocess.run(cmd, cwd=project_root, env=env).returncode
