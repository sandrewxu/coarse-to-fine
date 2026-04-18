"""Local latent generation (step 5) — vLLM / HF inference and output flattening."""

from src.generation.dataset import (
    flatten_for_c2f,
    load_prompts,
    save_generation_outputs,
    verify_and_filter_outputs,
)
from src.generation.inference import generate, generate_hf, generate_vllm

__all__ = [
    "flatten_for_c2f",
    "generate",
    "generate_hf",
    "generate_vllm",
    "load_prompts",
    "save_generation_outputs",
    "verify_and_filter_outputs",
]
