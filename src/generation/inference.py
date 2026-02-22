"""
vLLM-based inference for generating latent layer outputs from the SFT model.

The SFT model was trained on chat-format data where the user provides a 32-word
document and the assistant generates z_4: ...\nz_3: ...\nz_2: ...\nz_1: ... output.
This module replicates that format during generation.
"""
from typing import Any

from vllm import LLM, SamplingParams


class LatentGenerator:
    """vLLM-based generator for latent layer outputs."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vLLM engine from config.

        Args:
            config: Full experiment config; reads config['generation'] section.
        """
        gen_config = config["generation"]
        self.llm = LLM(
            model=gen_config["model_path"],
            tensor_parallel_size=gen_config.get("num_gpus", 1),
            seed=gen_config.get("seed", 42),
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=gen_config.get("max_tokens", 256),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", -1),
            repetition_penalty=gen_config.get("repetition_penalty", 1.0),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _build_chat_prompt(self, text: str) -> str:
        """
        Format a text prompt as a chat message matching the SFT training format.

        The SFT model was trained on chat completions where the user message
        is the 32-word document. We apply the tokenizer's chat template to
        produce the same format during generation.

        Args:
            text: The original document text to generate latents for.

        Returns:
            Formatted prompt string ready for the model.
        """
        messages = [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts: list[str]) -> list[str]:
        """
        Generate latent outputs for a batch of prompts.

        Each prompt is formatted as a chat message before generation.
        vLLM handles batching internally for maximum throughput.

        Args:
            prompts: List of original text documents.

        Returns:
            List of generated text strings (z_4: ...\nz_3: ... format).
        """
        formatted = [self._build_chat_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
