"""Shared helpers and resource bundle used by both RL reward managers.

Both ``C2FRewardManager`` (Phase A, q_φ training) and ``JointC2FRewardManager``
(joint posterior + decoder training) need to:

  - Load the frozen / trainable C2F decoder and its space tokenizer.
  - Parse latent layers from SFT-model output.
  - Build the flat C2F input sequence (BOS | z₄ | z₃ | z₂ | z₁ | x | pad).

This module factors those concerns out as pure functions plus a single resource
bundle (``C2FRewardComponents``) that each manager holds as ``self.components``.
The two reward classes then differ only in the *reward computation*, not in the
plumbing.
"""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from src.common.logging import get_logger
from src.verification import verify as verify_layers

if TYPE_CHECKING:
    # Heavy imports needed only by the load_c2f_* factories. Importing them
    # lazily lets ``src.rl.common`` be used from notebooks / unit tests
    # (test_reward_common.py) without pulling in transformers / huggingface.
    from src.c2f_model.modeling import C2FForCausalLM

log = get_logger(__name__)

# Default config path (relative to repo root); overridden by C2F_CONFIG_PATH env
# var in launchers that spawn workers with a stripped environment (notably veRL).
DEFAULT_CONFIG_PATH = "config/latent_generation.yaml"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

LayerNames = ("z_4", "z_3", "z_2", "z_1")
LabelStrategy = Literal["sft", "joint"]


# ── Pure helpers ─────────────────────────────────────────────────────────────


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning SFT outputs."""
    return _THINK_RE.sub("", text).strip()


def parse_layers(
    response: str,
    word_count_constraints: dict[str, int],
    *,
    strict: bool,
) -> list[str] | None:
    """Parse the four ``z_N: ...`` lines from an SFT response.

    Returns the layer content strings in ``[z_4, z_3, z_2, z_1]`` order, or
    ``None`` if the verifier rejects the structure.
    """
    cleaned = strip_think(response)
    result = verify_layers(cleaned, word_count_constraints, strict_word_count=strict)
    if not result.passed:
        return None
    return [layer.content for layer in result.layers]


def compute_word_boundaries(
    word_count_constraints: dict[str, int],
    text_word_count: int,
) -> list[tuple[int, int]]:
    """Compute (start, end) word indices for each scale block in the flat sequence.

    Layout is ``[z_4 words | z_3 words | z_2 words | z_1 words | text words]``.
    """
    word_counts = [word_count_constraints[name] for name in LayerNames]
    word_counts.append(text_word_count)
    boundaries: list[tuple[int, int]] = []
    pos = 0
    for wc in word_counts:
        boundaries.append((pos, pos + wc))
        pos += wc
    return boundaries


def build_c2f_input(
    layer_contents: list[str],
    prompt: str,
    *,
    scale_lengths: list[int],
    word_boundaries: list[tuple[int, int]],
    space_tokenizer: Any,
    bos_id: int,
    pad_id: int,
    seq_len: int,
    label_strategy: LabelStrategy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels) for one C2F sample.

    Replicates ``C2FDataset._build_token_sequence``: tokenises each scale's
    text segment to exactly ``scale_lengths[k]`` tokens (truncate or right-pad),
    prepends BOS, and right-pads the whole sequence to ``seq_len``.

    Args:
        layer_contents: ``[z_4, z_3, z_2, z_1]`` content strings.
        prompt: the original document text (the ``x`` tokens).
        label_strategy:
            - ``"sft"`` masks BOS and any ``pad_id`` token in labels (matches
              ``C2FDataset``'s unshifted-loss labels).
            - ``"joint"`` masks BOS and *all* tokens beyond ``1 + sum(scale_lengths)``
              (the post-content padding region), keeping the in-block padding live.
    """
    flat_parts = [*layer_contents, prompt]
    words = " ".join(flat_parts).split()

    tokens = [bos_id]
    for (start, end), length in zip(word_boundaries, scale_lengths, strict=False):
        segment_text = " ".join(words[start:end])
        encoded = space_tokenizer.encode(segment_text, add_special_tokens=False)
        if len(encoded) >= length:
            encoded = encoded[:length]
        else:
            encoded = encoded + [pad_id] * (length - len(encoded))
        tokens.extend(encoded)

    while len(tokens) < seq_len:
        tokens.append(pad_id)

    input_ids = torch.tensor(tokens[:seq_len], dtype=torch.long)
    labels = input_ids.clone()
    labels[0] = -100

    if label_strategy == "sft":
        labels[input_ids == pad_id] = -100
    elif label_strategy == "joint":
        content_len = 1 + sum(scale_lengths)
        labels[content_len:] = -100
    else:  # pragma: no cover — Literal type guards this
        raise ValueError(f"unknown label_strategy: {label_strategy!r}")

    return input_ids, labels


def load_c2f_weights(model: "C2FForCausalLM", checkpoint_path: Path) -> "C2FForCausalLM":
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


# ── Resource bundle + factory ────────────────────────────────────────────────


@dataclass
class C2FRewardComponents:
    """Bundle of resources shared by both reward managers.

    Each manager holds one of these as ``self.components``; phase-specific
    state (optimizer, save dir, locks, format-bonus weights) lives on the
    manager itself.
    """

    exp_config: dict
    scale_lengths: list[int]
    word_count_constraints: dict[str, int]
    text_word_count: int
    sft_tokenizer: Any
    c2f_model: "C2FForCausalLM"
    space_tokenizer: Any
    strict_word_count: bool
    bos_id: int
    pad_id: int
    seq_len: int
    word_boundaries: list[tuple[int, int]]


def load_exp_config() -> dict:
    """Load the experiment YAML, honouring ``C2F_CONFIG_PATH`` if set."""
    from src.config import load_config

    config_path = os.environ.get("C2F_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    return load_config(config_path)


def load_c2f_components(
    exp_config: dict,
    sft_tokenizer: Any,
    *,
    c2f_section_key: str,
    c2f_mask_type: str | None = None,
) -> C2FRewardComponents:
    """Build a ``C2FRewardComponents`` from the experiment YAML.

    Loads the C2F decoder weights and the space tokenizer, computes derived
    constants, and bundles them for the calling reward manager. Does **not**
    set device / eval-or-train mode / requires_grad — the caller decides those
    according to phase semantics (frozen for SFT, trainable for joint).

    Args:
        exp_config: full experiment dict from :func:`load_exp_config`.
        sft_tokenizer: HF tokenizer for the SFT model (provided by veRL at init).
        c2f_section_key: which RL subsection holds the C2F checkpoint path
            (``"sft_rl"`` or ``"joint"``).
        c2f_mask_type: if given, overrides ``mask_type`` on the loaded C2F
            config (joint phase uses ``"causal"``).
    """
    # Lazy imports — these pull in transformers / huggingface and would block
    # importing this module from notebooks or unit tests if hoisted to module level.
    from src.c2f_model.configuration import C2FConfig
    from src.c2f_model.modeling import C2FForCausalLM
    from src.c2f_model.training.tokenizer import load_or_train_space_tokenizer

    rl_section = exp_config.get("rl", {}).get(c2f_section_key, {})

    # ── C2F model ────────────────────────────────────────────────────────────
    c2f_checkpoint = Path(rl_section.get("c2f_model_path", "checkpoints/decoder"))
    suffix = f" (mask_type={c2f_mask_type})" if c2f_mask_type else ""
    log.info("Loading C2F from %s%s", c2f_checkpoint, suffix)
    model_config = C2FConfig.from_pretrained(str(c2f_checkpoint))
    if c2f_mask_type is not None:
        model_config.mask_type = c2f_mask_type
    c2f_model = C2FForCausalLM(model_config)
    c2f_model = load_c2f_weights(c2f_model, c2f_checkpoint)

    # ── Space tokenizer ──────────────────────────────────────────────────────
    c2f_train_cfg = exp_config.get("c2f_training", {})
    tokenizer_dir = Path(c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer"))
    log.info("Loading space tokenizer from %s", tokenizer_dir)
    space_tokenizer = load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=c2f_train_cfg.get("dataset_dir", "data/sft_dataset"),
        dataset_format=c2f_train_cfg.get("dataset_format", "sft"),
    )

    # ── Derived constants ────────────────────────────────────────────────────
    scale_lengths: list[int] = exp_config["scale_lengths"]
    word_count_constraints: dict[str, int] = exp_config["word_count_constraints"]
    text_word_count: int = exp_config.get("text_word_count", 32)
    strict_word_count: bool = exp_config.get("verification", {}).get("strict_word_count", True)

    bos_id = space_tokenizer.bos_token_id or space_tokenizer.eos_token_id
    pad_id = space_tokenizer.pad_token_id or space_tokenizer.eos_token_id
    seq_len = 2 ** math.ceil(math.log2(1 + sum(scale_lengths)))
    word_boundaries = compute_word_boundaries(word_count_constraints, text_word_count)

    return C2FRewardComponents(
        exp_config=exp_config,
        scale_lengths=scale_lengths,
        word_count_constraints=word_count_constraints,
        text_word_count=text_word_count,
        sft_tokenizer=sft_tokenizer,
        c2f_model=c2f_model,
        space_tokenizer=space_tokenizer,
        strict_word_count=strict_word_count,
        bos_id=bos_id,
        pad_id=pad_id,
        seq_len=seq_len,
        word_boundaries=word_boundaries,
    )
