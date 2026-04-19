"""Infrastructure constants.

These are *not* experiment knobs — they capture invariants of the runtime
(e.g. memory limits on a single H100, the canonical scale layout). Tunable
hyperparameters belong in ``config/latent_generation.yaml`` and the matching
Pydantic schema in ``src/config.py``.
"""

# vLLM memory budget — matches SFT max_length(256) + max_response(256) = 512.
# Qwen3-4B's native max_position_embeddings is 40960; leaving max_model_len
# unset makes vLLM reserve KV cache for that full context, which silently
# OOMs a single H100 (even 141 GiB) when colocated with an FSDP actor.
VLLM_MAX_MODEL_LEN: int = 512
VLLM_MAX_NUM_SEQS: int = 64

# Canonical scale layout used across the codebase. Override per-experiment via
# `scale_lengths` in the config YAML.
DEFAULT_SCALE_LENGTHS: tuple[int, ...] = (2, 4, 8, 16, 32)
