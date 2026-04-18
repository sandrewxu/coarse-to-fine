"""C2FConfig — Qwen3Config extended with the multi-scale sequence layout."""

import math

from transformers import Qwen3Config


class C2FConfig(Qwen3Config):
    r"""
    Configuration for the Coarse-to-Fine (C2F) language model.

    Extends :class:`Qwen3Config` with two new parameters that define the
    multi-scale sequence layout.  All other Qwen3 hyperparameters are
    inherited unchanged.

    The full context window is laid out as::

        [BOS | scale_0 tokens | scale_1 tokens | ... | scale_{K-1} tokens | padding]

    ``scale_lengths[-1]`` is the **text** scale (finest resolution);
    all earlier scales are latent summary representations.

    Args:
        scale_lengths (``list[int]``, *optional*, defaults to ``[2, 4, 8, 16, 32]``):
            Number of tokens per scale, ordered coarse → fine.  The last
            entry is treated as the text scale.
        **kwargs:
            Forwarded to :class:`Qwen3Config`.
    """

    # C2F: new model type so HF AutoModel dispatches to this class, not Qwen3.
    model_type = "c2f"

    def __init__(
        self,
        # C2F: coarse-to-fine scale layout; last entry is the text scale.
        scale_lengths: list[int] | None = None,
        # C2F: "block" = block-prefix mask (within-scale tokens independent),
        #       "causal" = standard lower-triangular autoregressive mask.
        mask_type: str = "block",
        **kwargs,
    ) -> None:
        # C2F: default scale layout matching the plan: 4 latent scales + text.
        if scale_lengths is None:
            scale_lengths = [2, 4, 8, 16, 32]
        self.scale_lengths = scale_lengths

        if mask_type not in ("block", "causal"):
            raise ValueError(f"mask_type must be 'block' or 'causal', got {mask_type!r}")
        self.mask_type = mask_type

        # C2F: the block-prefix mask is an arbitrary additive bias that Flash
        # Attention 2 does not support — FA2 ignores custom masks and silently
        # uses its own causal kernel, breaking the C2F attention pattern.
        # Force eager attention so the mask is always applied correctly.
        kwargs.setdefault("attn_implementation", "eager")
        super().__init__(**kwargs)

    # C2F: derived properties so callers never have to recompute these.

    @property
    def num_scale_levels(self) -> int:
        """Number of scales (latent + text)."""
        return len(self.scale_lengths)

    @property
    def seq_len(self) -> int:
        """
        Padded sequence length: nearest power of 2 to
        ``1 (BOS) + sum(scale_lengths)``.
        """
        total = 1 + sum(self.scale_lengths)
        return 2 ** math.ceil(math.log2(total))


__all__ = ["C2FConfig"]
