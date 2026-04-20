#!/usr/bin/env python3
"""
Train the masked-diffusion (MDLM) baseline.

Mirrors ``scripts/06b_train_ar_baseline.py`` but swaps the standard HF
``Trainer`` for ``DiffusionTrainer``, which implements the continuous-time
SUBS-MDLM loss from Sahoo et al. 2024 (``kuleshov-group/mdlm``).

The baseline reuses the AR baseline's dataset (``ARDataset``), tokenizer
(the C2F space tokenizer — ``[UNK]`` doubles as the MASK index), and 95/5
data split (``seed=42``). All three baselines (C2F, AR, diffusion) are
therefore directly comparable via ``scripts/09_eval_nll.py``.

Usage:
    python scripts/06c_train_diffusion_baseline.py \\
        --data data/local_generations/c2f_train.parquet \\
        --config config/H100_joint_causal.yaml

    accelerate launch --num_processes=4 scripts/06c_train_diffusion_baseline.py \\
        --data data/local_generations/c2f_train.parquet \\
        --config config/H100_joint_causal.yaml
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

    df = config["diffusion_training"]
    checkpoint_dir = Path(df.get("checkpoint_dir", "checkpoints/diffusion_baseline"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = df.get("eval_steps")
    report_to = "wandb" if wandb_enabled else df.get("report_to", "none")

    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=df.get("per_device_batch_size", 8),
        per_device_eval_batch_size=df.get("eval_batch_size") or df.get("per_device_batch_size", 8),
        gradient_accumulation_steps=df.get("gradient_accumulation_steps", 4),
        num_train_epochs=df.get("epochs", 1),
        learning_rate=df.get("lr", 5e-5),
        weight_decay=df.get("weight_decay", 0.01),
        warmup_ratio=df.get("warmup_ratio", 0.05),
        lr_scheduler_type=df.get("lr_scheduler_type", "cosine"),
        max_grad_norm=df.get("max_grad_norm", 1.0),
        logging_steps=df.get("logging_steps", 10),
        save_steps=df.get("save_steps", 500),
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        fsdp=df.get("fsdp", ""),
        fsdp_config=df.get("fsdp_config", {}),
        report_to=report_to,
        run_name=os.environ.get("WANDB_NAME") or df.get("run_name", "diffusion-baseline"),
        seed=df.get("seed", 42),
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


def _build_model(config, vocab_size):
    """Stock Qwen3ForCausalLM matching the diffusion_training arch overrides."""
    import math

    from transformers import Qwen3Config, Qwen3ForCausalLM

    df = config["diffusion_training"]
    kwargs = {"vocab_size": vocab_size}
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if df.get(key) is not None:
            kwargs[key] = df[key]
    text_words = config.get("text_word_count", 32)
    kwargs.setdefault("max_position_embeddings", 2 ** math.ceil(math.log2(1 + text_words)))
    # Eager attention is required: the FA2 kernel ignores 4D additive masks,
    # which would silently revert bidirectional attention to causal.
    kwargs.setdefault("attn_implementation", "eager")
    model_config = Qwen3Config(**kwargs)
    return Qwen3ForCausalLM(model_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MDLM-style masked-diffusion baseline")
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

    config = {
        "scale_lengths": [2, 4, 8, 16, 32],
        "diffusion_training": {},
        "c2f_training": {},
    }
    wandb_enabled = False

    if args.config:
        if not args.config.exists():
            log.error(f"Config not found: {args.config}")
            return 1
        from src.common.env import load_env, setup_wandb
        from src.config import load_config

        load_env()
        config = load_config(args.config)
        wandb_enabled = setup_wandb(config, step_name="diffusion-baseline")

    df_cfg = config["diffusion_training"]

    if args.epochs is not None:
        df_cfg["epochs"] = args.epochs
    if args.lr is not None:
        df_cfg["lr"] = args.lr
    if args.batch_size is not None:
        df_cfg["per_device_batch_size"] = args.batch_size
    if args.checkpoint_dir is not None:
        df_cfg["checkpoint_dir"] = str(args.checkpoint_dir)

    data_path = args.data.resolve()
    dataset_dir = data_path.parent
    parquet_filename = data_path.name
    dataset_format = detect_dataset_format(data_path)
    log.info(f"Dataset: {data_path} (format={dataset_format})")

    # Reuse the C2F space tokenizer for vocab parity at eval time.
    from src.c2f_model.training.dataset import ARDataset
    from src.c2f_model.training.diffusion import DiffusionTrainer
    from src.c2f_model.training.tokenizer import MASK_TOKEN, load_or_train_space_tokenizer

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
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    pad_id = tokenizer.pad_token_id
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Tokenizer does not contain {MASK_TOKEN!r} as a distinct id; "
            f"delete {tokenizer_dir} and retrain."
        )
    log.info(
        f"  Space tokenizer ready (vocab_size={vocab_size}, "
        f"mask_id={mask_id} [MASK], pad_id={pad_id})"
    )

    wandb_run = None
    if wandb_enabled and os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb

        wandb_run = wandb.init(
            name=df_cfg.get("run_name", "diffusion-baseline"),
            config={
                "step": "06c-diffusion-baseline",
                "diffusion_training": df_cfg,
                "scale_lengths": config.get("scale_lengths"),
            },
        )

    log.info("Building Qwen3 backbone (bidirectional, eager attention)...")
    model = _build_model(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {param_count:,}")

    log.info("Building dataset (text-only, masking applied in compute_loss)...")
    full_dataset = ARDataset(
        data_dir=str(dataset_dir),
        scale_lengths=config["scale_lengths"],
        text_word_count=config.get("text_word_count", 32),
        parquet_filename=parquet_filename,
        dataset_format=dataset_format,
        tokenizer=tokenizer,
    )

    eval_split = df_cfg.get("eval_split", 0.05)
    splits = full_dataset.train_test_split(test_size=eval_split, seed=df_cfg.get("seed", 42))
    log.info(f"  Train: {len(splits['train'])}, Eval: {len(splits['test'])}")

    training_args = _build_training_args(config, PROJECT_ROOT, wandb_enabled=wandb_enabled)
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        mask_id=mask_id,
        pad_id=pad_id,
        eps_t=df_cfg.get("eps_t", 1e-3),
        noise_eps=df_cfg.get("noise_eps", 1e-3),
        antithetic=df_cfg.get("antithetic", True),
    )

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    log.info(f"Model saved to: {training_args.output_dir}")

    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except Exception as e:  # pragma: no cover
        log.warning(f"Could not save tokenizer next to checkpoint: {e}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
