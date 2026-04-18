"""C2F model definition — Qwen3 with block-prefix attention and per-scale embeddings."""

from src.c2f_model.configuration import C2FConfig
from src.c2f_model.modeling import C2FForCausalLM, C2FModel

__all__ = ["C2FConfig", "C2FForCausalLM", "C2FModel"]
