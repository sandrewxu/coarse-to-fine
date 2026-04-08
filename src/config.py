"""
Experiment configuration schema and loader.

Defines Pydantic models for every config section with typed defaults.
The YAML file only needs to specify values that differ from defaults.
Derived fields (``word_count_constraints``, ``text_word_count``) are
computed automatically from ``scale_lengths``.

Usage::

    from src.config import load_config

    config = load_config("config/latent_generation.yaml")
    # config is a plain dict with all defaults filled in
"""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str | None = None
    entity: str | None = None
    group: str = "latent-generation"
    tags: list[str] = Field(default_factory=lambda: ["qwen3-4b"])
    mode: str = "online"


class VerificationConfig(BaseModel):
    strict_word_count: bool = True


class BatchConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5-nano-2025-08-07"
    reasoning_effort: str = "low"
    verbosity: str = "medium"
    system_prompt: str = "latent-generation"
    user_prompt: str = "gemini-3-pro-7"
    few_shot_examples: str = "latent-generation"
    run_tag: str = "latent_generation_10k_v1"
    prompt_data_dir: str = "data/prompt_data"
    output_dir: str = "data/batch_outputs"


class DataPrepConfig(BaseModel):
    dataset: str = "tinystoriesv2"
    memory_gb: float = 8.0
    seed: int = 42
    num_chunks: int = 4
    words_per_chunk: int = 32
    k_validation: int = 5000
    k_test: int = 5000
    k_prompt: int = 25000
    k_rl: int = 8000
    raw_data_dir: str = "data"


class DatasetConfig(BaseModel):
    output_dir: str = "data/verified/latent_generation_10k_v1"
    data_dir: str = "data/tinystoriesv2_shuffled"
    dataset_name: str = "tinystoriesv2"
    num_chunks: int = 4
    # Split filenames (relative to data_dir).
    # Named *_split to distinguish from sft.prompt_data (batch API request JSONL).
    val_split: str = "tinystoriesv2.val.jsonl"
    test_split: str = "tinystoriesv2.test.jsonl"
    prompt_split: str = "tinystoriesv2.prompt.jsonl"
    rl_split: str = "tinystoriesv2.rl.jsonl"


class SftConfig(BaseModel):
    model: str = "Qwen/Qwen3-4B"
    num_gpus: int = 1
    dataset_dir: str = "data/sft_dataset"
    checkpoint_dir: str = "checkpoints/sft"
    prompt_data: str = ""
    max_length: int = 256
    model_dtype: str = "bf16"
    micro_batch_size_per_gpu: int = 16
    use_remove_padding: bool = True
    use_liger: bool = True
    train_batch_size: int = 64
    epochs: int = 2
    lr: float = 1.0e-5


class GenerationConfig(BaseModel):
    model_path: str = ""
    output_dir: str = "data/local_generations"
    num_gpus: int = 1
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: int = 42
    verify_outputs: bool = True


class C2FTrainingConfig(BaseModel):
    init_from: str = "random"
    dataset_dir: str = "data/sft_dataset"
    dataset_format: str = "sft"
    tokenizer: str = "space"
    tokenizer_dir: str = "checkpoints/decoder/tokenizer"
    checkpoint_dir: str = "checkpoints/decoder"
    num_gpus: int = 1
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    lr: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_split: float = 0.05
    fsdp: str = "full_shard"
    report_to: str = "none"
    run_name: str = "c2f-pretrain"
    seed: int = 42


class RlSftConfig(BaseModel):
    model_path: str = ""
    c2f_model_path: str = "checkpoints/decoder"
    num_gpus: int = 1
    dataset_dir: str = "data/rl_dataset"
    checkpoint_dir: str = "checkpoints/rl/sft"
    rollout_n: int = 8
    max_prompt_length: int = 64
    max_response_length: int = 256
    train_batch_size: int = 64
    temperature: float = 1.0
    lr: float = 1.0e-6
    kl_coef: float = 0.01
    format_bonus_weight: float = 0.1
    ppo_micro_batch_size_per_gpu: int = 8
    dataloader_num_workers: int = 4
    epochs: int = 1


class C2fFinetuneConfig(BaseModel):
    model_path: str = "checkpoints/decoder"
    sft_model_path: str = "checkpoints/rl/sft"
    dataset_dir: str = "data/rl_dataset"
    generation_output_dir: str = "data/rl_dataset/c2f_finetune"
    checkpoint_dir: str = "checkpoints/rl/c2f"
    num_gpus: int = 1
    num_samples: int | None = None
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    lr: float = 5.0e-5
    epochs: int = 3
    fsdp: str = "full_shard"


class RlConfig(BaseModel):
    sft_rl: RlSftConfig = Field(default_factory=RlSftConfig)
    c2f_finetune: C2fFinetuneConfig = Field(default_factory=C2fFinetuneConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

_LATENT_LAYER_NAMES = ["z_4", "z_3", "z_2", "z_1"]


class ExperimentConfig(BaseModel):
    scale_lengths: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32])
    num_gpus: int = 1
    seed: int = 42

    wandb: WandbConfig = Field(default_factory=WandbConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    data_prep: DataPrepConfig = Field(default_factory=DataPrepConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sft: SftConfig = Field(default_factory=SftConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    c2f_training: C2FTrainingConfig = Field(default_factory=C2FTrainingConfig)
    rl: RlConfig = Field(default_factory=RlConfig)

    @property
    def word_count_constraints(self) -> dict[str, int]:
        """Derived from ``scale_lengths[:-1]`` -- not specified in YAML."""
        return dict(zip(_LATENT_LAYER_NAMES, self.scale_lengths[:-1]))

    @property
    def text_word_count(self) -> int:
        """The last entry of ``scale_lengths`` is the text (finest) scale."""
        return self.scale_lengths[-1]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_PROPAGATED_KEYS: list[tuple[str, ...]] = [
    # top-level num_gpus → every section
    ("sft", "num_gpus"),
    ("generation", "num_gpus"),
    ("c2f_training", "num_gpus"),
    ("rl", "sft_rl", "num_gpus"),
    ("rl", "c2f_finetune", "num_gpus"),
    # top-level seed → sections that have one
    ("data_prep", "seed"),
    ("generation", "seed"),
    ("c2f_training", "seed"),
]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config, validate it, and return a dict.

    Pydantic fills in defaults for any keys not present in the YAML.
    The returned dict also includes the derived ``word_count_constraints``
    and ``text_word_count`` keys for backward compatibility.

    Top-level ``num_gpus`` and ``seed`` are propagated to every section
    that has its own copy of those fields, so you only set them once.

    All existing ``config["key"]`` / ``.get()`` access patterns continue
    to work unchanged.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    validated = ExperimentConfig(**raw)
    result = validated.model_dump()
    result["word_count_constraints"] = validated.word_count_constraints
    result["text_word_count"] = validated.text_word_count

    for key_path in _PROPAGATED_KEYS:
        top_key = key_path[-1]  # e.g. "num_gpus" or "seed"
        node = result
        for key in key_path[:-1]:
            node = node[key]
        node[top_key] = result[top_key]

    return result
