"""
C2F reward manager for GRPO training (Phase A).

Reward = log p_θ(x, z) / num_tokens  (C2F log-joint, token-normalized)
       + format_bonus                 (format_bonus_weight when all 4 layers
                                       have the correct word count; partial
                                       credit proportional to passing layers)

Implements veRL's class-based reward manager interface:

    class MyRewardManager:
        def __init__(self, tokenizer, num_examine=0, ...): ...
        def __call__(self, data: DataProto, return_dict=False): ...

veRL provides the SFT model tokenizer at init so we can decode response token
IDs back to strings.  The C2F model and space tokenizer are loaded from paths
stored in the experiment YAML (``config/latent_generation.yaml``),
which is located via the ``C2F_CONFIG_PATH`` environment variable (set by the
SLURM/launch script before torchrun) or defaults to the repo-relative path.
"""
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.c2f_training.tokenizer import load_or_train_space_tokenizer
from src.qwen3_joint.configuration import C2FConfig
from src.qwen3_joint.modeling import C2FForCausalLM
from src.verification import verify as verify_layers

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Default config path (relative to repo root), overridden by C2F_CONFIG_PATH env var.
_DEFAULT_CONFIG_PATH = "config/latent_generation.yaml"


def _load_c2f_weights(model: C2FForCausalLM, checkpoint_path: Path) -> C2FForCausalLM:
    """Load weights from a saved checkpoint (safetensors or pytorch_model.bin)."""
    sf_path = checkpoint_path / "model.safetensors"
    pt_path = checkpoint_path / "pytorch_model.bin"

    if sf_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(sf_path))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found at {checkpoint_path}. "
            "Expected model.safetensors or pytorch_model.bin."
        )
    model.load_state_dict(state_dict)
    return model


class C2FRewardManager:
    """
    Reward manager for GRPO training on the SFT model q_φ.

    Reward = log p_θ(x, z) / num_tokens + format_bonus

    where:
      - log p_θ(x, z)  is the negative C2F cross-entropy loss scaled by the
                        number of unmasked tokens (so longer sequences are not
                        unfairly penalised).
      - format_bonus    is ``format_bonus_weight`` when every latent layer has
                        the correct word count, else proportional partial credit.

    The C2F model is kept frozen throughout; only q_φ is trained.

    Args:
        tokenizer:
            The SFT model tokenizer provided by veRL at init time.  Used to
            decode response token IDs back to strings.
        num_examine:
            veRL compatibility argument (unused).
        config_path:
            Path to the experiment YAML.  Defaults to the ``C2F_CONFIG_PATH``
            environment variable, or ``config/latent_generation.yaml``
            relative to the repository root.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        config_path: str | None = None,
        **kwargs,
    ):
        # ── Load experiment config ───────────────────────────────────────────
        from src.config import load_config

        if config_path is None:
            config_path = os.environ.get("C2F_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        self.config = load_config(config_path)

        self.scale_lengths: list[int] = self.config["scale_lengths"]
        self.word_count_constraints: dict[str, int] = self.config["word_count_constraints"]
        self.text_word_count: int = self.config.get("text_word_count", 32)

        rl_cfg = self.config.get("rl", {}).get("sft_rl", {})
        self.format_bonus_weight: float = float(rl_cfg.get("format_bonus_weight", 0.1))

        # ── SFT tokenizer (provided by veRL) ────────────────────────────────
        self.sft_tokenizer = tokenizer

        # ── Frozen C2F model ─────────────────────────────────────────────────
        c2f_checkpoint = rl_cfg.get("c2f_model_path", "checkpoints/decoder")
        c2f_checkpoint = Path(c2f_checkpoint)
        print(f"[C2FRewardManager] Loading frozen C2F from {c2f_checkpoint}...")
        model_config = C2FConfig.from_pretrained(str(c2f_checkpoint))
        self.c2f_model = C2FForCausalLM(model_config)
        self.c2f_model = _load_c2f_weights(self.c2f_model, c2f_checkpoint)
        self.c2f_model.eval()
        for p in self.c2f_model.parameters():
            p.requires_grad_(False)
        self.device = next(self.c2f_model.parameters()).device

        # ── Space tokenizer (word-level, 1 word = 1 token) ──────────────────
        c2f_train_cfg = self.config.get("c2f_training", {})
        tokenizer_dir = Path(
            c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
        )
        print(f"[C2FRewardManager] Loading space tokenizer from {tokenizer_dir}...")
        self.space_tokenizer = load_or_train_space_tokenizer(
            tokenizer_dir=tokenizer_dir,
            data_dir=c2f_train_cfg.get("dataset_dir", "data/sft_dataset"),
            dataset_format=c2f_train_cfg.get("dataset_format", "sft"),
        )

        # ── Verification config ─────────────────────────────────────────────
        self.strict_word_count = self.config.get("verification", {}).get("strict_word_count", True)

        # ── Token ID shortcuts ───────────────────────────────────────────────
        self.bos_id = (
            self.space_tokenizer.bos_token_id or self.space_tokenizer.eos_token_id
        )
        self.pad_id = (
            self.space_tokenizer.pad_token_id or self.space_tokenizer.eos_token_id
        )
        self.seq_len = 2 ** math.ceil(math.log2(1 + sum(self.scale_lengths)))

        # ── Word boundaries for flat sequence segmentation ───────────────────
        # Layout: [z_4 words | z_3 words | z_2 words | z_1 words | text words]
        layer_names = ["z_4", "z_3", "z_2", "z_1"]
        word_counts = [self.word_count_constraints[n] for n in layer_names]
        word_counts.append(self.text_word_count)
        self.word_boundaries: list[tuple[int, int]] = []
        pos = 0
        for wc in word_counts:
            self.word_boundaries.append((pos, pos + wc))
            pos += wc

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _strip_think(self, text: str) -> str:
        """Strip <think>...</think> blocks from SFT output."""
        return _THINK_RE.sub("", text).strip()

    def _parse_layers(self, response: str) -> list[str] | None:
        """
        Parse z_n: lines from SFT response (after stripping think blocks).

        Returns layer content strings in [z_4, z_3, z_2, z_1] order, or None
        if parsing or verification fails.
        """
        cleaned = self._strip_think(response)
        result = verify_layers(
            cleaned, self.word_count_constraints,
            strict_word_count=self.strict_word_count,
        )
        if not result.passed:
            return None
        return [layer.content for layer in result.layers]

    def _format_bonus(self, response: str) -> float:
        """
        Compute format bonus.

        Full bonus (``format_bonus_weight``) when all 4 layers pass; partial
        credit proportional to the number of correctly-sized layers otherwise.
        """
        cleaned = self._strip_think(response)
        result = verify_layers(
            cleaned, self.word_count_constraints,
            strict_word_count=self.strict_word_count,
        )
        if result.passed:
            return self.format_bonus_weight

        # Partial credit even when overall verification fails
        if result.layers:
            expected_layers = ["z_4", "z_3", "z_2", "z_1"]
            correct = sum(
                1
                for layer in result.layers
                if layer.word_count == self.word_count_constraints.get(layer.layer_name, -1)
            )
            return self.format_bonus_weight * (correct / len(expected_layers))
        return 0.0

    def _build_c2f_input(
        self, layer_contents: list[str], prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build C2F input_ids and labels from layer content strings + prompt.

        Replicates the tokenisation logic from
        ``C2FDataset._build_token_sequence`` / ``_build_labels``.

        Returns:
            (input_ids, labels) tensors of shape [seq_len].
        """
        flat_parts = layer_contents + [prompt]
        words = " ".join(flat_parts).split()

        tokens = [self.bos_id]
        for (start, end), length in zip(self.word_boundaries, self.scale_lengths):
            segment_text = " ".join(words[start:end])
            encoded = self.space_tokenizer.encode(segment_text, add_special_tokens=False)
            if len(encoded) >= length:
                encoded = encoded[:length]
            else:
                encoded = encoded + [self.pad_id] * (length - len(encoded))
            tokens.extend(encoded)

        while len(tokens) < self.seq_len:
            tokens.append(self.pad_id)

        input_ids = torch.tensor(tokens[: self.seq_len], dtype=torch.long)

        # Labels: mask BOS and padding (unshifted loss, same as C2FDataset)
        labels = input_ids.clone()
        labels[0] = -100
        labels[input_ids == self.pad_id] = -100

        return input_ids, labels

    @torch.no_grad()
    def _log_p_c2f(self, layer_contents: list[str], prompt: str) -> tuple[float, int]:
        """
        Compute log p_θ(x, z) via a forward pass of the frozen C2F model.

        Returns:
            (log_prob, num_unmasked_tokens)
            log_prob = -CE_loss × num_unmasked  (unnormalised log-likelihood)
        """
        input_ids, labels = self._build_c2f_input(layer_contents, prompt)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        outputs = self.c2f_model(input_ids=input_ids, labels=labels)
        loss: float = outputs.loss.item()
        num_unmasked: int = int((labels != -100).sum().item())
        log_p = -loss * num_unmasked
        return log_p, num_unmasked

    # ── veRL interface ───────────────────────────────────────────────────────

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """
        Compute rewards for a batch of GRPO rollouts.

        Expects:
            data.batch['responses']          – response token IDs (B × T_resp)
            data.non_tensor_batch['ground_truth'] – original document per sample

        Sets:
            data.batch['token_level_scores'] – reward placed at last real token
                                               position, zeros elsewhere (B × T_resp)

        Returns:
            DataProto with token_level_scores populated (or a dict if
            ``return_dict=True``).
        """
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        # Decode response token IDs → strings
        response_strs: list[str] = self.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )

        # Original documents (set as ground_truth during RL parquet prep)
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        reward_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32)
        pad_id = getattr(self.sft_tokenizer, "pad_token_id", None)

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths)):
            # 1. Parse latent layers from SFT output
            layer_contents = self._parse_layers(response)
            if layer_contents is None:
                # Malformed output: zero reward
                continue

            # 2. log p_θ(x, z) from frozen C2F
            log_p, num_tokens = self._log_p_c2f(layer_contents, str(prompt))
            log_p_normalized = log_p / max(num_tokens, 1)

            # 3. Format bonus
            bonus = self._format_bonus(response)

            reward = log_p_normalized + bonus

            # 4. Place reward at last non-padding response token
            response_ids = data.batch["responses"][i]
            if pad_id is not None:
                non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
            else:
                non_pad = torch.arange(response_len, device=response_ids.device)
            last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
            reward_tensor[i, last_pos] = reward

        data.batch["token_level_scores"] = reward_tensor

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data
