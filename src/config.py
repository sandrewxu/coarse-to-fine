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
    num_chunks: int = 8
    words_per_chunk: int = 32
    k_validation: int = 5000
    k_test: int = 5000
    k_prompt: int = 25000
    k_rl: int = 8000
    raw_data_dir: str = "data"


class DatasetConfig(BaseModel):
    # Script 03 appends ``batch.run_tag`` to this at write time, so the final
    # stats path is ``{output_dir}/{run_tag}/verification_stats.json``.
    output_dir: str = "data/verified"
    data_dir: str = "data/tinystoriesv2_shuffled"
    dataset_name: str = "tinystoriesv2"
    num_chunks: int = 8
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
    seed: int = 42


class GenerationConfig(BaseModel):
    model_path: str = ""
    output_dir: str = "data/local_generations"
    chunks: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
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
    dataset_dir: str = "data/local_generations"
    dataset_format: str = "sft"
    tokenizer: str = "space"
    tokenizer_dir: str = "checkpoints/tokenizer"
    checkpoint_dir: str = "checkpoints/decoder"
    num_gpus: int = 1
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    per_device_batch_size: int = 8
    eval_batch_size: int | None = None  # falls back to per_device_batch_size
    gradient_accumulation_steps: int = 4
    epochs: int = 1
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
    mask_type: str = "block"


class DiffusionTrainingConfig(BaseModel):
    """MDLM-faithful masked-diffusion baseline (continuous-time SUBS).

    Reads the same parquet and reuses the same space tokenizer as
    ``c2f_training`` / ``ar_training``. The MASK index is the tokenizer's
    ``[UNK]`` (id=1) — see ``src/c2f_model/training/diffusion.py``.
    """

    dataset_dir: str = "data/local_generations"
    dataset_format: str = "sft"
    checkpoint_dir: str = "checkpoints/diffusion_baseline"
    num_gpus: int = 1
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    per_device_batch_size: int = 8
    eval_batch_size: int | None = None  # falls back to per_device_batch_size
    gradient_accumulation_steps: int = 4
    epochs: int = 1
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
    run_name: str = "diffusion-baseline"
    seed: int = 42
    # MDLM-specific knobs:
    eps_t: float = 1e-3
    noise_eps: float = 1e-3
    antithetic: bool = True
    n_eval_samples: int = 128


class ARTrainingConfig(BaseModel):
    """No-latents autoregressive baseline matched to ``c2f_training`` for fair comparison.

    Reads the same parquet, uses the same space tokenizer
    (``c2f_training.tokenizer_dir``), but trains a stock ``Qwen3ForCausalLM``
    on just the text portion of each document with shifted-CE.
    """

    dataset_dir: str = "data/local_generations"
    dataset_format: str = "sft"
    checkpoint_dir: str = "checkpoints/ar_baseline"
    num_gpus: int = 1
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    per_device_batch_size: int = 8
    eval_batch_size: int | None = None  # falls back to per_device_batch_size
    gradient_accumulation_steps: int = 4
    epochs: int = 1
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
    run_name: str = "ar-baseline"
    seed: int = 42


class RlSftConfig(BaseModel):
    model_path: str = ""
    c2f_model_path: str = "checkpoints/decoder"
    num_gpus: int = 1
    dataset_dir: str = "data/rl_dataset"
    checkpoint_dir: str = "checkpoints/rl/sft"
    rollout_n: int = 8
    max_prompt_length: int = 256
    max_response_length: int = 256
    train_batch_size: int = 64
    temperature: float = 1.0
    lr: float = 1.0e-6
    # ── KL-to-reference regularizer (separate from ELBO) ────────────────────
    # This is veRL's ``actor.use_kl_loss`` / ``kl_loss_coef`` knob, a stability
    # regularizer ``+ coef · KL(q_φ ∥ q_ref)`` on the actor loss. It is NOT
    # part of the ELBO — the ELBO's -log q_φ term lives in the entropy bonus
    # (Option B) or in the reward's -log q_ref term (Option A, opt-in). Off by
    # default: pure REINFORCE-on-ELBO. Flip to True only as an ablation.
    use_kl_loss: bool = False
    kl_coef: float = 0.0
    format_bonus_weight: float = 0.1
    # ── ELBO entropy term (preferred path, "Option B") ──────────────────────
    # veRL's actor loss subtracts ``entropy_coeff · H(q_φ)`` from the loss,
    # equivalently adding it to the maximization objective. At
    # ``entropy_coeff = 1.0`` the PPO objective becomes
    #   E_q[log p_θ(x, z)] + H(q_φ) = ELBO
    # exactly (up to the usual PPO ratio approximation). Uses the full-
    # distribution per-token entropy, which is a lower-variance estimator
    # than the sampled ``-log q_φ(z)`` MC term. Default 1.0 = turn the ELBO on.
    entropy_coeff: float = 1.0
    # ── Explicit reward-side -log q_ref term ("Option A", opt-in) ──────────
    # Adds ``-ref_nll_coef · log q_ref(z|x)`` to the scalar reward by loading
    # a frozen reference SFT model inside the reward manager. At ``α = 1``
    # paired with ``kl_coef = 1.0`` this is mathematically equivalent to
    # ``entropy_coeff = 1.0`` — but uses a single-sample MC estimate (higher
    # variance) and an extra 8 GB of GPU memory. Kept as an ablation / debug
    # path. Default 0.0: OFF. Set to 1.0 only if you want to A/B against the
    # entropy-bonus formulation or log per-rollout ``log q_ref`` for analysis.
    ref_nll_coef: float = 0.0
    # Path to the reference SFT model. If empty, falls back to ``model_path``
    # (the actor's initial checkpoint — the natural reference since the actor
    # *starts* as this model). Only loaded when ``ref_nll_coef > 0``.
    ref_model_path: str = ""
    ppo_micro_batch_size_per_gpu: int = 8
    dataloader_num_workers: int = 4
    epochs: int = 1
    # vLLM rollout knobs — propagated to veRL as ``actor_rollout_ref.rollout.*``
    # in ``src/rl/verl_config.py``. Must be declared here so YAML values aren't
    # silently dropped by Pydantic's default extra='ignore' behaviour.
    rollout_gpu_memory_utilization: float = 0.6
    rollout_max_num_seqs: int = 256
    rollout_max_num_batched_tokens: int = 32768
    # Rollout parallelism split: TP × DP must equal num_gpus. TP=1 is right for
    # small actors (<=13B) where TP all-reduces dominate per-layer compute.
    # DP=None → num_gpus // tp (full data-parallel rollout).
    rollout_tensor_parallel_size: int = 1
    rollout_data_parallel_size: int | None = None


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


class JointConfig(BaseModel):
    model_path: str = ""
    c2f_model_path: str = "checkpoints/decoder"
    c2f_lr: float = 1e-4
    c2f_weight_decay: float = 0.01
    # Save the C2F decoder every N RL steps (matches trainer.save_freq semantics
    # for the actor). Implementation in src/rl/reward_joint.py converts to a
    # per-sample threshold using train_batch_size × rollout_n.
    c2f_save_steps: int = 50
    c2f_save_dir: str = "checkpoints/rl/joint/c2f"
    c2f_mask_type: str = "causal"
    num_gpus: int = 1
    dataset_dir: str = "data/rl_dataset"
    checkpoint_dir: str = "checkpoints/rl/joint"
    max_prompt_length: int = 256
    max_response_length: int = 256
    train_batch_size: int = 256
    save_freq: int = 150
    temperature: float = 1.0
    lr: float = 1e-6
    malformed_reward: float = -10.0
    ppo_micro_batch_size_per_gpu: int = 16
    dataloader_num_workers: int = 4
    epochs: int = 12
    c2f_micro_batch_size: int = 32
    c2f_keep_last_n: int = 2
    # ELBO entropy term — see RlSftConfig.entropy_coeff for the derivation.
    # Default 1.0 adds H(q_φ) to the maximization objective, giving the ELBO
    # entropy term exactly.
    entropy_coeff: float = 1.0
    # vLLM rollout knobs — propagated to veRL as ``actor_rollout_ref.rollout.*``
    # in ``src/rl/verl_config.py``. Must be declared here so YAML values aren't
    # silently dropped by Pydantic's default extra='ignore' behaviour.
    rollout_gpu_memory_utilization: float = 0.6
    rollout_max_num_seqs: int = 256
    rollout_max_num_batched_tokens: int = 32768
    # Number of rollout samples per prompt. n=1 is plain REINFORCE++; n>1
    # enables a within-prompt baseline (GRPO-style) which reduces advantage
    # variance, the standard fix for the policy gradient being drowned by the
    # entropy bonus under entropy_coeff=1.0. Total sequences per step =
    # train_batch_size × rollout_n, so step time scales with rollout_n.
    rollout_n: int = 1
    # Rollout parallelism split: TP × DP must equal num_gpus. TP=1 is right for
    # small actors (<=13B) where TP all-reduces dominate per-layer compute.
    # DP=None → num_gpus // tp (full data-parallel rollout).
    rollout_tensor_parallel_size: int = 1
    rollout_data_parallel_size: int | None = None
    # veRL trainer.resume_mode. "auto" (default) picks up from the latest
    # checkpoint in checkpoint_dir; "disable" starts fresh. Set to "disable"
    # for throughput sweeps where each probe must be independent — otherwise
    # global_step accumulates across probes and total_training_steps becomes
    # ambiguous.
    resume_mode: str = "auto"
    # Advantage estimator for veRL. "grpo" computes per-prompt baselines (good
    # when rollout_n>1); "reinforce_plus_plus" uses a global batch-wide baseline.
    # See verl/trainer/ppo/core_algos.py for the per-estimator math.
    adv_estimator: str = "grpo"
    # Rollout-debug knobs read by JointC2FRewardManager.run_single. The first
    # ``debug_dump_initial`` samples per worker get logged unconditionally,
    # then every ``debug_dump_every``-th sample after that. Set
    # ``debug_dump_every=0`` to disable the post-warmup cadence.
    debug_dump_initial: int = 10
    debug_dump_every: int = 1000


class RlConfig(BaseModel):
    sft_rl: RlSftConfig = Field(default_factory=RlSftConfig)
    c2f_finetune: C2fFinetuneConfig = Field(default_factory=C2fFinetuneConfig)
    joint: JointConfig = Field(default_factory=JointConfig)


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
    ar_training: ARTrainingConfig = Field(default_factory=ARTrainingConfig)
    diffusion_training: DiffusionTrainingConfig = Field(default_factory=DiffusionTrainingConfig)
    rl: RlConfig = Field(default_factory=RlConfig)

    @property
    def word_count_constraints(self) -> dict[str, int]:
        """Derived from ``scale_lengths[:-1]`` -- not specified in YAML."""
        return dict(zip(_LATENT_LAYER_NAMES, self.scale_lengths[:-1], strict=False))

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
    ("ar_training", "num_gpus"),
    ("diffusion_training", "num_gpus"),
    ("rl", "sft_rl", "num_gpus"),
    ("rl", "c2f_finetune", "num_gpus"),
    ("rl", "joint", "num_gpus"),
    # top-level seed → sections that have one
    ("data_prep", "seed"),
    ("generation", "seed"),
    ("c2f_training", "seed"),
    ("ar_training", "seed"),
    ("diffusion_training", "seed"),
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
