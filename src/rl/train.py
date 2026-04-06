"""
ELBO optimisation training functions for Step 7.

Phase A — ``run_sft_rl()``:
    GRPO on q_φ (SFT model), p_θ (C2F) frozen.
    Gradient of φ: REINFORCE with reward = log p_θ(x, z).
    KL(q_φ ‖ q_φ^ref) is handled by veRL via actor.use_kl_loss=true.

Phase B — ``run_c2f_finetune()``:
    Supervised fine-tuning of p_θ (C2F), q_φ (SFT) frozen.
    Gradient of θ: E_{q_φ}[∇_θ log p_θ(x, z)].
    Reuses existing step-5 and step-6 infrastructure unchanged.
"""
import sys
import subprocess
from pathlib import Path
from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────


def _prep_rl_parquet(config: dict[str, Any], project_root: Path) -> Path:
    """
    Build the RL training parquet from the SFT parquet.

    Copies the SFT parquet and adds two columns required by veRL GRPO:
      - ``ground_truth``: copy of ``prompt`` (the original document), used by
        the reward manager to run the C2F forward pass.
      - ``data_source``: literal ``"latent_generation"``.

    Returns:
        Path to the saved RL parquet (skips regeneration if it already exists).
    """
    from datasets import load_dataset, Dataset

    rl_sft_cfg = config["rl"]["sft_rl"]
    sft_dataset_dir = Path(config.get("sft", {}).get("dataset_dir", "data/sft_dataset"))
    if not sft_dataset_dir.is_absolute():
        sft_dataset_dir = project_root / sft_dataset_dir

    rl_dataset_dir = Path(rl_sft_cfg.get("dataset_dir", "data/rl_dataset"))
    if not rl_dataset_dir.is_absolute():
        rl_dataset_dir = project_root / rl_dataset_dir
    rl_dataset_dir.mkdir(parents=True, exist_ok=True)

    sft_parquet = sft_dataset_dir / "train.parquet"
    rl_parquet = rl_dataset_dir / "sft_rl.parquet"

    if rl_parquet.exists():
        print(f"  RL parquet already exists: {rl_parquet}")
        return rl_parquet

    if not sft_parquet.exists():
        raise FileNotFoundError(
            f"SFT parquet not found: {sft_parquet}\n"
            "Run step 3 (03_verify_outputs.py) first."
        )

    print(f"  Preparing RL parquet from {sft_parquet}...")
    ds = load_dataset("parquet", data_files=str(sft_parquet), split="train")

    def _add_rl_columns(row):
        return {
            "prompt": row["prompt"],
            "response": row["response"],
            "ground_truth": row["prompt"],
            "data_source": "latent_generation",
        }

    ds = ds.map(_add_rl_columns, remove_columns=ds.column_names)
    ds.to_parquet(str(rl_parquet))
    print(f"  Saved RL parquet: {rl_parquet} ({len(ds):,} samples)")
    return rl_parquet


# ── Phase A ───────────────────────────────────────────────────────────────────


def run_sft_rl(
    config: dict[str, Any],
    project_root: Path,
    *,
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

    from src.rl.verl_config import build_verl_grpo_overrides, get_verl_grpo_entrypoint

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

    entrypoint = get_verl_grpo_entrypoint()
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m",
        entrypoint,
        *overrides,
    ]
    print("Phase A — GRPO on q_φ:")
    print("  Command:", " ".join(cmd))

    # Export config path for the reward manager
    _venv = os.environ.get("VIRTUAL_ENV") or str(Path(sys.executable).parent.parent)
    env = {
        **os.environ,
        "C2F_CONFIG_PATH": str(project_root / "config/experiments/latent_generation.yaml"),
        "UV_NO_SYNC": "1",
        "UV_PROJECT_ENVIRONMENT": _venv,
    }
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
         (Uses :class:`src.generation.inference.LatentGenerator`.)
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
    from src.generation.inference import LatentGenerator

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
    # Override model_path and num_gpus in the generation config
    gen_config = dict(config.get("generation", {}))
    gen_config["model_path"] = str(sft_model_path)
    gen_config["num_gpus"] = int(
        c2f_ft_cfg.get("num_gpus", gen_config.get("num_gpus", 1))
    )
    modified_config = {**config, "generation": gen_config}

    generator = LatentGenerator(modified_config)
    outputs: list[str] = generator.generate(prompts)

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
