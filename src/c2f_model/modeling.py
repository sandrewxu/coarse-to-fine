"""C2FForCausalLM — Qwen3 forked for coarse-to-fine generation.

Three differences from upstream `transformers.models.qwen3.modeling_qwen3`:

1. **Per-scale absolute embeddings** (`C2FScaleEmbedding`) replace RoPE — token
   position is determined by which scale block it belongs to, not its index.
2. **Block-prefix attention mask** (`create_c2f_block_causal_mask`) replaces the
   causal mask — tokens at scale k attend to all coarser scales but are
   independent within the same scale.
3. **Unshifted cross-entropy** loss — labels align 1:1 with input positions
   rather than predicting the next token.

Every line that diverges from upstream Qwen3 is annotated with `# C2F:`.
To upgrade the base model, diff against `transformers.models.qwen3.modeling_qwen3`
and re-apply each `# C2F:` block. See `README.md` in this folder for details.
"""

# C2F: -----------------------------------------------------------------------
# All code below is specific to the Coarse-to-Fine model.  Every departure
# from the Qwen3 base is annotated with a "# C2F:" comment.
# ----------------------------------------------------------------------------
from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

from src.c2f_model.configuration import C2FConfig

# Compat: these decorators/types were added in transformers 4.52; provide
# no-op fallbacks so the model works with 4.51 (shipped by verl / vllm).
try:
    from transformers.utils import TransformersKwargs, auto_docstring
except ImportError:
    TransformersKwargs = FlashAttentionKwargs

    def auto_docstring(fn):
        return fn


try:
    from transformers.utils.generic import merge_with_config_defaults
except ImportError:

    def merge_with_config_defaults(fn):
        return fn


try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:

    def capture_outputs(fn):
        return fn


def create_c2f_block_causal_mask(
    config,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``[batch, 1, seq_len, seq_len]`` additive attention mask.

    Token *i* (belonging to scale *k_i*) may attend to token *j* iff
    ``scale(j) < scale(i)``.  The BOS token at position 0 is assigned
    scale ``-1`` so every content token can attend to it.  Content tokens
    do **not** self-attend (to avoid leaking target identity through the
    residual stream when using unshifted loss).  BOS and padding positions
    self-attend to prevent all-``-inf`` softmax rows.

    ``0.0`` = attend, ``-inf`` = masked.
    """
    # C2F: assign a scale index to every position in the padded sequence.
    #   BOS  →  scale -1   (universal prefix)
    #   scale k tokens  →  scale k   (0 … num_scale_levels-1)
    #   padding  →  scale num_scale_levels   (never attended to or from)
    num_scales = len(config.scale_lengths)
    scale_of = torch.full((seq_len,), fill_value=num_scales, dtype=torch.long, device=device)
    scale_of[0] = -1  # BOS
    pos = 1
    for k, length in enumerate(config.scale_lengths):
        end = min(pos + length, seq_len)
        scale_of[pos:end] = k
        pos += length

    # C2F: attend[i, j] = True  iff  scale(j) < scale(i)  (j is a strictly
    # earlier scale than i, so i may condition on j).
    attend = scale_of.unsqueeze(0) < scale_of.unsqueeze(1)  # [seq, seq]

    # C2F: BOS (scale -1) attends to nothing via the strict-less-than rule,
    # so it needs a self-attend entry to avoid an all-(-inf) softmax row.
    # Content tokens always attend to at least BOS, so they do NOT get
    # self-attend — otherwise the unshifted loss leaks the answer (the
    # token embedding at position i is the target for logits[i]).
    attend[0, 0] = True  # BOS self-attend only

    # C2F: nothing should attend to padding, and padding should not attend out.
    is_pad = scale_of == num_scales
    attend[:, is_pad] = False  # mask padding as key
    attend[is_pad, :] = False  # mask padding as query (re-add diagonal below)
    # Give padding rows a self-attend to avoid NaN in unused positions.
    pad_idx = is_pad.nonzero(as_tuple=True)[0]
    attend[pad_idx, pad_idx] = True

    # C2F: convert boolean attend → additive float mask (0 / -inf).
    mask = torch.full(
        (seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=device, dtype=dtype
    )
    mask[attend] = 0.0

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def create_causal_mask(
    config,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``[batch, 1, seq_len, seq_len]`` standard causal attention mask.

    Position *i* attends to all positions *j* ≤ *i* (standard autoregressive).
    Padding positions (beyond the content region) are fully masked, with
    self-attend on the diagonal to avoid NaN softmax rows.

    ``0.0`` = attend, ``-inf`` = masked.
    """
    # Standard lower-triangular: attend[i, j] = (j <= i)
    attend = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Mask padding positions (same logic as block mask)
    content_len = 1 + sum(config.scale_lengths)  # BOS + all scales
    is_pad = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_pad[content_len:] = True

    attend[:, is_pad] = False  # nothing attends to padding
    attend[is_pad, :] = False  # padding does not attend out
    pad_idx = is_pad.nonzero(as_tuple=True)[0]
    attend[pad_idx, pad_idx] = True  # self-attend to avoid NaN

    mask = torch.full(
        (seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=device, dtype=dtype
    )
    mask[attend] = 0.0

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


class C2FScaleEmbedding(nn.Module):
    """Per-scale learned absolute position embeddings.

    For each of the ``num_scale_levels`` scales, an independent
    ``nn.Embedding`` table of size ``(scale_lengths[k], hidden_size)`` is
    trained.  The BOS token at position 0 gets its own learned vector.
    Padding positions receive a zero vector.

    ``forward`` returns a ``[batch, seq_len, hidden_size]`` tensor that is
    added to the token embeddings before the transformer layers.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # C2F: one independent embedding table per scale.
        self.scale_lengths = config.scale_lengths
        self.full_seq_len = config.seq_len
        hidden_size = config.hidden_size

        self.embeddings = nn.ModuleList(
            [nn.Embedding(length, hidden_size) for length in config.scale_lengths]
        )
        # C2F: learned position vector for the BOS token.
        self.bos_emb = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build and return ``[batch_size, seq_len, hidden_size]`` position tensor."""
        dtype = self.bos_emb.dtype
        hidden_size = self.bos_emb.shape[0]

        # C2F: concatenate BOS embedding, per-scale embeddings, and zero padding.
        parts: list[torch.Tensor] = [self.bos_emb.unsqueeze(0)]  # [1, H]
        for emb, length in zip(self.embeddings, self.scale_lengths, strict=False):
            indices = torch.arange(length, device=device)
            parts.append(emb(indices))  # [length, H]

        total_content = 1 + sum(self.scale_lengths)
        pad_len = self.full_seq_len - total_content
        if pad_len > 0:
            parts.append(torch.zeros(pad_len, hidden_size, device=device, dtype=dtype))

        pos_emb = torch.cat(parts, dim=0)  # [full_seq_len, H]
        pos_emb = pos_emb[:seq_len]  # C2F: clip to the actual input length
        return pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, H]


class C2FAttention(Qwen3Attention):
    """Qwen3Attention without Rotary Position Embeddings.

    Identical to :class:`Qwen3Attention` except the two RoPE lines are
    removed from ``forward``.  All other behaviour (GQA, QK-norm,
    sliding-window support, flash-attention dispatch) is inherited.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[
            torch.Tensor, torch.Tensor
        ],  # C2F: kept for API compat, value is (None, None)
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # C2F: removed — cos, sin = position_embeddings
        # C2F: removed — query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # C2F: removed sin/cos from cache_kwargs (not used without RoPE)
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class C2FDecoderLayer(Qwen3DecoderLayer):
    """Qwen3DecoderLayer using :class:`C2FAttention` instead of Qwen3Attention.

    Adds an optional ``residual_override`` kwarg to ``forward``. When provided,
    the post-attention residual connection uses ``residual_override`` instead
    of the layer's own ``hidden_states``. This is used at layer 0 in block mode
    to split the residual path (masked, to prevent leak of ``input_ids[i]``
    into ``logits[i]``) from the K/V path (real embeddings, so earlier-scale
    attention actually carries token info). Default ``None`` preserves the
    upstream Qwen3DecoderLayer behavior bit-for-bit.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        # C2F: replace Qwen3Attention with C2FAttention (no RoPE)
        self.self_attn = C2FAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        residual_override: torch.Tensor | None = None,  # C2F: block-mode residual split
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # C2F: mirror Qwen3DecoderLayer.forward, but allow the post-attention
        # residual to come from ``residual_override``. K/V still use the real
        # ``hidden_states`` via input_layernorm → self_attn.
        residual = hidden_states if residual_override is None else residual_override
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # C2F: second residual uses post-attn hidden_states as normal — at this
        # point the masked residual is already baked in, so no override needed.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class C2FModel(Qwen3Model):
    """Coarse-to-Fine transformer backbone.

    Differences from :class:`Qwen3Model`:

    1. Uses :class:`C2FDecoderLayer` (no RoPE inside attention).
    2. Adds per-scale absolute position embeddings via
       :class:`C2FScaleEmbedding` instead of rotary embeddings.
    3. Uses a block-prefix causal mask (see :func:`create_c2f_block_causal_mask`)
       instead of a standard lower-triangular causal mask.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        # C2F: replace all decoder layers with C2F versions
        self.layers = nn.ModuleList(
            [C2FDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # C2F: remove rotary embedding; position information comes from scale embeddings
        del self.rotary_emb

        # C2F: per-scale learned absolute position embeddings
        self.scale_pos_emb = C2FScaleEmbedding(config)

        # C2F: learned mask embedding for block mode. Used ONLY on the layer-0
        # post-attention residual at content positions; real token embeddings
        # still feed Q/K/V at every layer so cross-scale attention carries real
        # per-example info. With the strict block mask (no self-attend, no
        # same-scale attend for content), attn_out[i] at layer 0 has no
        # dependence on input_ids[i]; the masked residual then breaks the
        # residual-stream leak that unshifted CE would otherwise expose.
        # See C2FModel.forward for the full construction.
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size))

        # C2F: number of content positions (BOS + all scales) for masking logic
        self._content_len = 1 + sum(config.scale_lengths)

        # C2F: cache for the block-prefix mask, keyed by (seq_len, device, dtype).
        # The mask is deterministic given config+shape+dtype, so there is no need
        # to recompute it on every forward pass.
        self._mask_cache: dict[tuple, torch.Tensor] = {}

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # C2F: track whether the caller provided pre-built embeddings (inference
        # with actual token context for already-generated scales).  If so, skip
        # block-mode masking since the caller is responsible for the content.
        caller_provided_embeds = inputs_embeds is not None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # C2F: per-scale absolute position embeddings replace RoPE. Computed once
        # and added to both the real embedding stream (used as Q/K/V input) and the
        # block-mode masked residual built below.
        scale_pos = self.scale_pos_emb(
            batch_size=inputs_embeds.shape[0],
            seq_len=inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        inputs_embeds = inputs_embeds + scale_pos

        # C2F: block mode uses unshifted CE (logits[i] predicts labels[i] = input_ids[i])
        # which would leak via the residual stream if input_emb(input_ids[i]) reached
        # the LM head at position i. We kill that leak by overriding the layer-0
        # post-attention residual at content positions with mask_embedding +
        # scale_pos_emb[i]. K/V at every layer still see real embeddings so cross-scale
        # attention actually carries per-example information (an earlier fix clobbered
        # the embeddings before the transformer, which killed the leak but also
        # destroyed K/V content for earlier-scale attention — collapsing training loss
        # to per-position unigram).
        #
        # The block attention mask strictly forbids self-attend and same-scale attend
        # for content tokens, so attn_out[i] at layer 0 has no dependence on
        # input_ids[i]; combined with a masked residual the layer-0 output at i is
        # leak-free, and the property propagates by induction through later layers.
        #
        # Callers that pass ``inputs_embeds`` directly (e.g. inference with encoder
        # embeddings for already-generated scales) are responsible for their own leak
        # handling; we do not touch the residual in that case.
        block_residual_override: torch.Tensor | None = None
        if self.config.mask_type == "block" and not caller_provided_embeds:
            if past_key_values is not None:
                raise ValueError(
                    "C2F block mode does not support past_key_values without "
                    "caller-provided inputs_embeds. Block-mode inference should "
                    "supply scale embeddings explicitly via inputs_embeds."
                )
            content_end = min(self._content_len, inputs_embeds.shape[1])
            mask_vec = self.mask_embedding.to(inputs_embeds.dtype)  # (H,)
            block_residual_override = inputs_embeds.clone()
            # Content-position residual: mask_vec + scale_pos_emb[pos]. BOS and
            # padding positions inherit their real embedding (labels are -100 there).
            block_residual_override[:, 1:content_end] = (
                mask_vec + scale_pos[:, 1:content_end]
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # C2F: replace the standard causal mask with the block-prefix mask.
        # The standard mask is not valid for multi-scale prediction.
        #
        # The mask is the same for every batch at a given (seq_len, device, dtype),
        # so we cache it to avoid reallocating on every forward pass.
        #
        # Both "full_attention" and "sliding_window_attention" layer types get the
        # same C2F mask.  Qwen3 models larger than 4B use sliding-window attention
        # on some layers; without this key those models would raise a KeyError.
        seq_len = inputs_embeds.shape[1]
        mask_key = (seq_len, inputs_embeds.device, inputs_embeds.dtype)
        if mask_key not in self._mask_cache:
            mask_fn = (
                create_c2f_block_causal_mask
                if self.config.mask_type == "block"
                else create_causal_mask
            )
            self._mask_cache[mask_key] = mask_fn(
                config=self.config,
                seq_len=seq_len,
                batch_size=1,  # stored without batch dim; expanded below
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
        c2f_mask = self._mask_cache[mask_key].expand(inputs_embeds.shape[0], 1, seq_len, seq_len)
        causal_mask_mapping = {
            "full_attention": c2f_mask,
            "sliding_window_attention": c2f_mask,  # C2F: needed for Qwen3 models with SWA layers
        }

        hidden_states = inputs_embeds
        # C2F: no rotary embeddings; pass (None, None) to satisfy the layer call signature
        position_embeddings = (None, None)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # C2F: layer 0 in block mode swaps in the masked residual; see the
            # block_residual_override construction above for the rationale.
            layer_residual_override = block_residual_override if layer_idx == 0 else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                residual_override=layer_residual_override,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class C2FForCausalLM(Qwen3ForCausalLM):
    """Coarse-to-Fine language model with an LM head.

    Uses :class:`C2FModel` as the backbone and overrides the loss computation
    to use **unshifted** cross-entropy.  In the standard autoregressive LM,
    ``logits[i]`` predicts ``labels[i+1]``.  In C2F, the block-prefix mask
    means tokens within the same scale do **not** attend to each other, so
    each position independently predicts its own token given coarser-scale
    context.  Therefore ``logits[i]`` predicts ``labels[i]`` (no shift).
    """

    config_class = C2FConfig

    def __init__(self, config) -> None:
        # C2F: skip Qwen3ForCausalLM.__init__ to avoid constructing an unused
        # Qwen3Model.  Call the PreTrainedModel base directly instead.
        Qwen3PreTrainedModel.__init__(self, config)
        # C2F: use C2FModel instead of Qwen3Model
        self.model = C2FModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # C2F: override forward to use unshifted loss (logits[i] predicts labels[i]).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if self.config.mask_type == "causal":
                # C2F: SHIFTED cross-entropy — logits[i] predicts labels[i+1].
                # Standard autoregressive: each position predicts the next token.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                )
            else:
                # C2F: UNSHIFTED cross-entropy — logits[i] predicts labels[i].
                # Block-prefix mask: within-scale tokens are independent.
                loss = loss_fct(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1),
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = ["C2FForCausalLM", "C2FModel"]
