#!/usr/bin/env python3
"""
Supervised fine-tuning of q_φ (Qwen3-4B) on verified batch API outputs.

Trains the model to map documents to latent hierarchies using the chat format:
  user: <document>  →  assistant: z_4: ...\nz_3: ...\nz_2: ...\nz_1: ...

Saves standard HuggingFace checkpoints (model.safetensors) that vLLM and
downstream steps can load directly.

Usage:
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --config config/latent_generation.yaml
    python scripts/04_sft_train.py --data data/sft_dataset/train.parquet --num-gpus 2 --epochs 3

    # Multi-GPU with FSDP:
    accelerate launch --num_processes=2 scripts/04_sft_train.py \
        --data data/sft_dataset/train.parquet --config config/latent_generation.yaml
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training with HuggingFace Trainer")
    parser.add_argument("--data", required=True, type=Path, help="Training parquet (columns: prompt, response)")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}", file=sys.stderr)
        return 1

    # Load config
    from src.utils.env import load_env, setup_wandb
    load_env()

    sft_config: dict = {}
    wandb_enabled = False
    if args.config:
        if not args.config.exists():
            print(f"Error: Config not found: {args.config}", file=sys.stderr)
            return 1
        from src.config import load_config
        config = load_config(args.config)
        sft_config = config.get("sft", {})
        wandb_enabled = setup_wandb(config, step_name="sft")

    # CLI overrides
    model_name = sft_config.get("model", "Qwen/Qwen3-4B")
    max_length = sft_config.get("max_length", 256)
    epochs = args.epochs or sft_config.get("epochs", 2)
    lr = args.lr or sft_config.get("lr", 1e-5)
    per_device_batch = args.batch_size or sft_config.get("micro_batch_size_per_gpu", 16)
    train_batch_size = sft_config.get("train_batch_size", 64)
    checkpoint_dir = args.checkpoint_dir or Path(sft_config.get("checkpoint_dir", "checkpoints/sft"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Gradient accumulation to reach effective batch size
    num_gpus = args.num_gpus or sft_config.get("num_gpus", 1)
    grad_accum = max(1, train_batch_size // (per_device_batch * num_gpus))

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Load dataset
    print(f"Loading data: {args.data}")
    ds = load_dataset("parquet", data_files=str(args.data.resolve()), split="train")

    val_path = args.data.parent / "val.parquet"
    eval_ds = None
    if val_path.exists():
        eval_ds = load_dataset("parquet", data_files=str(val_path.resolve()), split="train")
        print(f"  Train: {len(ds)}, Val: {len(eval_ds)}")
    else:
        print(f"  Train: {len(ds)} (no val split found)")

    # Tokenize: format as chat (user=prompt, assistant=response), mask user tokens
    def tokenize(examples):
        all_input_ids = []
        all_labels = []

        for prompt, response in zip(examples["prompt"], examples["response"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            # Full conversation
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

            # Prompt-only (to find where response starts)
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

            # Labels: mask prompt tokens with -100, only train on response
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            all_input_ids.append(full_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels}

    print("Tokenizing...")
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    # Data collator with padding
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    # Training args
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
        run_name="sft-qwen3-4b",
        fsdp="full_shard" if num_gpus > 1 else "",
        seed=42,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"Starting SFT training: {epochs} epochs, batch={train_batch_size} (per_device={per_device_batch} x {grad_accum} accum x {num_gpus} gpu)")
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"Model saved to: {checkpoint_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
