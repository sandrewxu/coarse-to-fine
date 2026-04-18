"""Infrastructure constants.

These are *not* experiment knobs — they capture invariants of the runtime
(e.g. memory limits on a single H100, the canonical scale layout). Tunable
hyperparameters belong in ``config/latent_generation.yaml`` and the matching
Pydantic schema in ``src/config.py``.
"""

# vLLM memory budget — matches SFT max_length(256) + max_response(256) × 2 with
# headroom; >1024 over-allocates KV cache for Qwen3-4B in bf16 on a single H100.
VLLM_MAX_MODEL_LEN: int = 1024
VLLM_MAX_NUM_SEQS: int = 1024

# Canonical scale layout used across the codebase. Override per-experiment via
# `scale_lengths` in the config YAML.
DEFAULT_SCALE_LENGTHS: tuple[int, ...] = (2, 4, 8, 16, 32)
