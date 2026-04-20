#!/usr/bin/env python3
"""
Train the no-latents autoregressive (AR) baseline.

Mirrors ``scripts/06_train_decoder.py`` but uses a stock ``Qwen3ForCausalLM``
on the document text only — no latents, no scale embeddings, no block mask.

The AR baseline must be trained with the SAME space tokenizer
(``checkpoints/decoder/tokenizer``) and SAME 95/5 split (``seed=42``) as the
C2F model so that ``scripts/09_eval_nll.py`` can directly compare
``nats_per_word`` between the two checkpoints.

Usage:
    # Standard run via the experiment YAML:
    python scripts/06b_train_ar_baseline.py \\
        --data data/local_generations/c2f_train.parquet \\
        --config config/latent_generation.yaml

    # Multi-GPU with accelerate:
    accelerate launch --num_processes=4 scripts/06b_train_ar_baseline.py \\
        --data data/local_generations/c2f_train.parquet \\
        --config config/latent_generation.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.common.logging import get_logger

log = get_logger(__name__)

from src.common.paths import PROJECT_ROOT


def detect_dataset_format(parquet_path: Path) -> str:
    columns = pq.read_schema(str(parquet_path)).names
    if "text" in columns:
        return "c2f"
    if "prompt" in columns and "response" in columns:
        return "sft"
    raise ValueError(
        f"Cannot detect format from columns {columns}. "
        "Expected 'text' (c2f) or 'prompt'+'response' (sft)."
    )


def _build_training_args(config, project_root, *, wandb_enabled):
    import torch
    from transformers import TrainingArguments

    ar = config["ar_training"]
    checkpoint_dir = Path(ar.get("checkpoint_dir", "checkpoints/ar_baseline"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = ar.get("eval_steps")
    report_to = "wandb" if wandb_enabled else ar.get("report_to", "none")

    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=ar.get("per_device_batch_size", 8),
        per_device_eval_batch_size=ar.get("eval_batch_size") or ar.get("per_device_batch_size", 8),
        gradient_accumulation_steps=ar.get("gradient_accumulation_steps", 4),
        num_train_epochs=ar.get("epochs", 1),
        learning_rate=ar.get("lr", 5e-5),
        weight_decay=ar.get("weight_decay", 0.01),
        warmup_ratio=ar.get("warmup_ratio", 0.05),
        lr_scheduler_type=ar.get("lr_scheduler_type", "cosine"),
        max_grad_norm=ar.get("max_grad_norm", 1.0),
        logging_steps=ar.get("logging_steps", 10),
        save_steps=ar.get("save_steps", 500),
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        fsdp=ar.get("fsdp", ""),
        fsdp_config=ar.get("fsdp_config", {}),
        report_to=report_to,
        run_name=os.environ.get("WANDB_NAME") or ar.get("run_name", "ar-baseline"),
        seed=ar.get("seed", 42),
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


def _build_model(config, vocab_size):
    """Stock Qwen3ForCausalLM with the same architecture overrides as c2f_training."""
    import math

    from transformers import Qwen3Config, Qwen3ForCausalLM

    ar = config["ar_training"]
    kwargs = {"vocab_size": vocab_size}
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if ar.get(key) is not None:
            kwargs[key] = ar[key]
    # Cap RoPE buffers to the actual sequence length (BOS + text words).
    # Qwen3's 40960 default would over-allocate for our 33-token sequences.
    text_words = config.get("text_word_count", 32)
    kwargs.setdefault("max_position_embeddings", 2 ** math.ceil(math.log2(1 + text_words)))
    model_config = Qwen3Config(**kwargs)
    return Qwen3ForCausalLM(model_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train no-latents AR baseline")
    parser.add_argument("--data", required=True, type=Path, help="Training parquet file")
    parser.add_argument(
        "--config", type=Path, default=None, help="Experiment YAML for defaults and W&B"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from checkpoint directory"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        log.error(f"Data file not found: {args.data}")
        return 1

    config = {"scale_lengths": [2, 4, 8, 16, 32], "ar_training": {}, "c2f_training": {}}
    wandb_enabled = False

    if args.config:
        if not args.config.exists():
            log.error(f"Config not found: {args.config}")
            return 1
        from src.common.env import load_env, setup_wandb
        from src.config import load_config

        load_env()
        config = load_config(args.config)
        wandb_enabled = setup_wandb(config, step_name="ar-baseline")

    ar_cfg = config["ar_training"]

    if args.epochs is not None:
        ar_cfg["epochs"] = args.epochs
    if args.lr is not None:
        ar_cfg["lr"] = args.lr
    if args.batch_size is not None:
        ar_cfg["per_device_batch_size"] = args.batch_size
    if args.checkpoint_dir is not None:
        ar_cfg["checkpoint_dir"] = str(args.checkpoint_dir)

    data_path = args.data.resolve()
    dataset_dir = data_path.parent
    parquet_filename = data_path.name
    dataset_format = detect_dataset_format(data_path)
    log.info(f"Dataset: {data_path} (format={dataset_format})")

    # The AR baseline reuses the C2F space tokenizer for vocab parity at eval time.
    from transformers import Trainer

    from src.c2f_model.training.dataset import ARDataset
    from src.c2f_model.training.tokenizer import load_or_train_space_tokenizer

    c2f_cfg = config.get("c2f_training", {})
    tokenizer_dir = Path(c2f_cfg.get("tokenizer_dir", "checkpoints/tokenizer"))
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = PROJECT_ROOT / tokenizer_dir
    tokenizer = load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=dataset_dir,
        dataset_format=dataset_format,
        parquet_filename=parquet_filename,
    )
    vocab_size = tokenizer.vocab_size
    log.info(f"  Space tokenizer ready (vocab_size={vocab_size})")

    wandb_run = None
    if wandb_enabled and os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb

        wandb_run = wandb.init(
            name=ar_cfg.get("run_name", "ar-baseline"),
            config={
                "step": "06b-ar-baseline",
                "ar_training": ar_cfg,
                "scale_lengths": config.get("scale_lengths"),
            },
        )

    log.info("Building Qwen3 AR model...")
    model = _build_model(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {param_count:,}")

    log.info("Building dataset (text-only)...")
    full_dataset = ARDataset(
        data_dir=str(dataset_dir),
        scale_lengths=config["scale_lengths"],
        text_word_count=config.get("text_word_count", 32),
        parquet_filename=parquet_filename,
        dataset_format=dataset_format,
        tokenizer=tokenizer,
    )

    eval_split = ar_cfg.get("eval_split", 0.05)
    splits = full_dataset.train_test_split(test_size=eval_split, seed=ar_cfg.get("seed", 42))
    log.info(f"  Train: {len(splits['train'])}, Eval: {len(splits['test'])}")

    training_args = _build_training_args(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
    )

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    log.info(f"Model saved to: {training_args.output_dir}")

    # Save tokenizer alongside the model so eval_ar can find a vocab match.
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except Exception as e:  # pragma: no cover
        log.warning(f"Could not save tokenizer next to checkpoint: {e}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
