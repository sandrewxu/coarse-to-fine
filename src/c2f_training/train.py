"""
C2F model pretraining with HuggingFace Trainer + FSDP.

Handles model initialization (from SFT checkpoint, base model, or random),
weight transfer for compatible parameters, and TrainingArguments construction
from the experiment YAML config.
"""
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.qwen3_joint.configuration import C2FConfig
from src.qwen3_joint.modeling import C2FForCausalLM


def load_c2f_model(config: dict[str, Any]) -> C2FForCausalLM:
    """
    Load and initialize C2FForCausalLM.

    Handles three init modes:
    1. "random": Initialize from scratch with C2FConfig defaults.
    2. Path or HF model name: Load compatible Qwen3 weights, transfer to C2F.
       New C2F-specific parameters (scale_pos_emb, bos_emb) are randomly initialized.
       Rotary embedding weights from the source are dropped.

    Args:
        config: Full experiment config with 'c2f_training' and 'scale_lengths'.

    Returns:
        Initialized C2FForCausalLM model.
    """
    c2f_config = config["c2f_training"]
    init_from = c2f_config.get("init_from", "random")
    scale_lengths = config["scale_lengths"]

    if init_from == "random":
        model_config = C2FConfig(scale_lengths=scale_lengths)
        return C2FForCausalLM(model_config)

    # Load source config and create C2F config from it
    model_config = C2FConfig.from_pretrained(
        init_from,
        scale_lengths=scale_lengths,
    )
    model = C2FForCausalLM(model_config)

    # Transfer compatible weights from source checkpoint
    model = _load_compatible_weights(model, init_from)
    return model


def _load_compatible_weights(
    model: C2FForCausalLM, checkpoint_path: str
) -> C2FForCausalLM:
    """
    Load weights from a Qwen3/SFT checkpoint into the C2F model.

    Transfers all parameters that match by name and shape. New C2F-specific
    parameters (scale_pos_emb, bos_emb) remain randomly initialized.
    Rotary embedding weights are silently skipped.

    Args:
        model: Target C2FForCausalLM with randomly initialized weights.
        checkpoint_path: Path to source checkpoint or HF model name.

    Returns:
        Model with compatible weights loaded.
    """
    source = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    source_state = source.state_dict()
    target_state = model.state_dict()

    loaded, skipped = [], []
    for name, param in source_state.items():
        if name in target_state and target_state[name].shape == param.shape:
            target_state[name] = param
            loaded.append(name)
        else:
            skipped.append(name)

    model.load_state_dict(target_state)
    print(f"Weight transfer: loaded {len(loaded)}, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped params (new or shape mismatch): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    return model


def build_training_args(
    config: dict[str, Any], project_root: Path, *, wandb_enabled: bool = False
) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from the c2f_training config section.

    When ``wandb_enabled`` is True (set by ``setup_wandb`` in the calling script),
    ``report_to`` is forced to ``"wandb"`` regardless of what the YAML says.
    WANDB_* environment variables are already configured by ``setup_wandb``.

    Args:
        config: Full experiment config.
        project_root: Project root for resolving relative paths.
        wandb_enabled: Whether the top-level ``wandb.enabled`` flag is True.

    Returns:
        Configured TrainingArguments.
    """
    c2f = config["c2f_training"]

    checkpoint_dir = Path(c2f.get("checkpoint_dir", "checkpoints/decoder"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = c2f.get("eval_steps")

    # W&B: if enabled globally, override report_to regardless of YAML value
    report_to = "wandb" if wandb_enabled else c2f.get("report_to", "none")

    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=c2f.get("per_device_batch_size", 8),
        gradient_accumulation_steps=c2f.get("gradient_accumulation_steps", 4),
        num_train_epochs=c2f.get("epochs", 10),
        learning_rate=c2f.get("lr", 5e-5),
        weight_decay=c2f.get("weight_decay", 0.01),
        warmup_ratio=c2f.get("warmup_ratio", 0.05),
        lr_scheduler_type=c2f.get("lr_scheduler_type", "cosine"),
        max_grad_norm=c2f.get("max_grad_norm", 1.0),
        logging_steps=c2f.get("logging_steps", 10),
        save_steps=c2f.get("save_steps", 500),
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        fsdp=c2f.get("fsdp", ""),
        fsdp_config=c2f.get("fsdp_config", {}),
        report_to=report_to,
        run_name=c2f.get("run_name", "c2f-pretrain"),
        seed=c2f.get("seed", 42),
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
