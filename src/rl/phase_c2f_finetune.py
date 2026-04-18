"""Phase B — supervised fine-tuning of p_θ (C2F) using samples from frozen q_φ.

Implements ``∇_θ L = E_{q_φ}[∇_θ log p_θ(x, z)]``:

  1. Load prompts from the SFT training parquet.
  2. Generate ``z ~ q_φ(·|x)`` with the RL-updated SFT checkpoint (frozen here).
  3. Verify and filter, then flatten into a C2F training parquet.
  4. Fine-tune the C2F model on that parquet via :class:`C2FTrainer`.
"""

from pathlib import Path
from typing import Any

from src.common.logging import get_logger

log = get_logger(__name__)


def run_c2f_finetune(
    config: dict[str, Any],
    project_root: Path,
    *,
    wandb_enabled: bool = False,
) -> int:
    """Phase B: supervised gradient on p_θ using frozen-q_φ samples."""
    # Heavy imports (HF Trainer, vLLM) are lazy so the module can be imported
    # in CI without the GPU stack.
    from datasets import load_dataset

    from src.c2f_training.dataset import C2FDataset
    from src.c2f_training.tokenizer import load_or_train_space_tokenizer
    from src.c2f_training.train import C2FTrainer, build_training_args, load_c2f_model
    from src.generation.dataset import flatten_for_c2f, verify_and_filter_outputs
    from src.generation.inference import generate

    c2f_ft_cfg = config.get("rl", {}).get("c2f_finetune", {})
    if not c2f_ft_cfg:
        log.error("config['rl']['c2f_finetune'] section is missing.")
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
        log.error("SFT model not found: %s", sft_model_path)
        log.error("Run Phase A (--phase sft) first.")
        return 1

    # ── Step 1: load prompts ─────────────────────────────────────────────────
    sft_parquet = sft_dataset_dir / "train.parquet"
    if not sft_parquet.exists():
        log.error("SFT parquet not found: %s", sft_parquet)
        return 1

    ds = load_dataset("parquet", data_files=str(sft_parquet), split="train")
    prompts: list[str] = list(ds["prompt"])
    num_samples = c2f_ft_cfg.get("num_samples")
    if num_samples is not None:
        prompts = prompts[: int(num_samples)]
    log.info("Phase B — Step 1: Generating z ~ q_φ for %d prompts...", len(prompts))

    # ── Step 2: generate z ~ q_φ(·|x) with frozen SFT ───────────────────────
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

    # ── Step 3: verify and flatten ──────────────────────────────────────────
    log.info("Phase B — Step 2: Verifying and flattening outputs...")
    filtered_prompts, filtered_outputs, stats = verify_and_filter_outputs(prompts, outputs, config)
    log.info("%s", stats)

    if not filtered_prompts:
        log.error("No valid outputs after verification.")
        return 1

    c2f_parquet_path = flatten_for_c2f(
        filtered_prompts,
        filtered_outputs,
        output_dir=generation_output_dir,
        filename="c2f_finetune.parquet",
    )
    log.info("Saved %d samples to %s", len(filtered_prompts), c2f_parquet_path)

    # ── Step 4: fine-tune C2F ───────────────────────────────────────────────
    log.info("Phase B — Step 3: Fine-tuning C2F model on q_φ samples...")

    c2f_train_cfg = config.get("c2f_training", {})
    tokenizer_dir = Path(c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer"))
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
    log.info("Phase B complete. C2F model saved to: %s", training_args.output_dir)
    return 0
