"""SFT training (step 4) — fine-tune q_φ (Qwen3-4B) on verified batch outputs.

Trains the model to map documents to latent hierarchies via the chat format::

  user:      <document>
  assistant: z_4: ...
             z_3: ...
             z_2: ...
             z_1: ...

Saves a standard HuggingFace checkpoint that vLLM and downstream steps can
load directly.
"""

import os
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

log = get_logger(__name__)


def train_sft(
    config: dict[str, Any],
    project_root: Path,
    *,
    data_path: Path,
    wandb_enabled: bool = False,
    resume_from: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> int:
    """Run supervised fine-tuning of q_φ on the verified batch outputs.

    Args:
        config: Full experiment config (uses ``config["sft"]``).
        project_root: Repo root for resolving relative checkpoint paths.
        data_path: Path to the SFT training parquet (columns: prompt, response).
        wandb_enabled: True if W&B is configured (gates ``report_to="wandb"``).
        resume_from: Optional checkpoint dir to resume training from.
        cli_overrides: Optional dict of CLI overrides; recognised keys are
            ``epochs``, ``lr``, ``per_device_batch_size``, ``checkpoint_dir``,
            ``num_gpus``. Each overrides the matching config field if not None.

    Returns:
        0 on success, 1 if the data path is missing.
    """
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        return 1

    overrides = cli_overrides or {}
    sft_config = config.get("sft", {})

    model_name: str = sft_config.get("model", "Qwen/Qwen3-4B")
    max_length: int = sft_config.get("max_length", 256)
    epochs: int = overrides.get("epochs") or sft_config.get("epochs", 2)
    lr: float = overrides.get("lr") or sft_config.get("lr", 1e-5)
    per_device_batch: int = overrides.get("per_device_batch_size") or sft_config.get(
        "micro_batch_size_per_gpu", 4
    )
    train_batch_size: int = sft_config.get("train_batch_size", 16)
    seed: int = sft_config.get("seed", 42)

    checkpoint_dir = overrides.get("checkpoint_dir") or Path(
        sft_config.get("checkpoint_dir", "checkpoints/sft")
    )
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_gpus: int = overrides.get("num_gpus") or sft_config.get("num_gpus", 2)
    grad_accum: int = max(1, train_batch_size // (per_device_batch * num_gpus))

    # Heavy imports are deferred so the module is importable without GPU deps.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    log.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    log.info("Loading data: %s", data_path)
    train_ds = load_dataset("parquet", data_files=str(data_path.resolve()), split="train")

    val_path = data_path.parent / "val.parquet"
    eval_ds = None
    if val_path.exists():
        eval_ds = load_dataset("parquet", data_files=str(val_path.resolve()), split="train")
        log.info("  Train: %d, Val: %d", len(train_ds), len(eval_ds))
    else:
        log.info("  Train: %d (no val split found)", len(train_ds))

    def _tokenize(examples):
        """Format as user/assistant chat, mask user tokens with -100."""
        all_input_ids = []
        all_labels = []
        for prompt, response in zip(examples["prompt"], examples["response"], strict=False):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            all_input_ids.append(full_ids)
            all_labels.append(labels)
        return {"input_ids": all_input_ids, "labels": all_labels}

    log.info("Tokenizing...")
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=3,
        report_to="wandb" if wandb_enabled else "none",
        run_name=os.environ.get("WANDB_NAME") or "sft-qwen3-4b",
        gradient_checkpointing=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    log.info(
        "Starting SFT training: %d epochs, batch=%d (per_device=%d x %d accum x %d gpu)",
        epochs,
        train_batch_size,
        per_device_batch,
        grad_accum,
        num_gpus,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    # Cap max_position_embeddings on the saved config so downstream consumers
    # (vLLM rollout in step 7, HF generation in step 5) don't inherit
    # Qwen3-4B's 40960 default and over-reserve KV cache.
    model.config.max_position_embeddings = max_length
    trainer.save_model()
    tokenizer.save_pretrained(str(checkpoint_dir))
    log.info("Model saved to: %s", checkpoint_dir)
    return 0
