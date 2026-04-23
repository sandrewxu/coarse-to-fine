"""Sample K valid ``z`` responses from ``q_φ`` for IWAE-K evaluation.

The ELBO identity used by :func:`src.eval.bound.eval_c2f_bound` is

    log p(x) >= E_{z ~ q(z|x)} [log p(x, z) - log q(z|x)]

For K=1 we cheat by reading the gold ``z`` from the test parquet — that ``z``
was itself drawn from ``q_φ`` at batch-API time, so it's a valid (albeit old)
sample. For K>1 we need fresh samples from ``q_φ`` at eval time. This module
provides that.

Two concerns the sampler has to handle:

1. **Validity.** C2F requires each latent scale to be exactly
   ``word_count_constraints[layer]`` words. ``q_φ`` is *trained* to produce
   valid outputs, but not guaranteed to. We rejection-sample: draw >K,
   keep only the valid ones, re-draw for docs that came up short. If a doc
   cannot reach K valid draws within ``max_tries_per_doc`` attempts, we
   raise loudly rather than silently report a biased bound.

2. **Scoring ``log q(z|x)``.** Rather than intercept ``output_scores`` from
   ``model.generate`` (subtle under temperature scaling + EOS truncation),
   we defer to the existing :func:`src.eval.bound._sft_nll_per_doc`, which
   teacher-forces the full response and sums CE over response tokens only.
   Costs one extra forward pass per sample; at eval time that's fine.

The rejection-sampling correction term ``log q(valid | x)`` is dropped. For
a well-trained ``q_φ`` on strictly-verified SFT data the validity rate is
typically >95 % so ``log q(valid | x) ≈ 0``; including it would require a
separate MC estimate per doc. If the empirical validity rate on your data
drops below ~90 %, the bound starts to look optimistic by roughly
``-log q(valid|x)`` nats per doc, and you should either (a) estimate the
correction explicitly, (b) switch to constrained decoding (reject at the
token level inside generate), or (c) retrain ``q_φ``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.common.logging import get_logger
from src.rl.common import LayerNames, parse_layers

log = get_logger(__name__)


@dataclass
class QSample:
    """One valid ``(response_text, parsed_layers)`` draw from ``q_φ``."""

    response: str
    layers: list[str]  # [z_4, z_3, z_2, z_1] content strings


def _apply_sft_prompt(tokenizer, prompt: str) -> str:
    """Chat-template the user-prompt side exactly as SFT training does."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _generate_raw_responses(
    model,
    tokenizer,
    prompts: list[str],
    num_return_sequences: int,
    *,
    temperature: float,
    max_new_tokens: int,
    device: torch.device,
) -> list[list[str]]:
    """Return ``[[response_1, ..., response_S], ...]`` (one inner list per prompt).

    Uses ``model.generate`` with ``num_return_sequences`` draws per prompt.
    ``do_sample=True`` so the draws are genuinely random. Left-pads the batch
    (required for correct generation with padded prompts).

    We don't care about the per-token log-probs from generate here — the
    downstream scorer re-runs teacher forcing to get ``log q(z | x)``.
    """
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    # Left-pad: HF generate expects left-padded inputs for batched sampling.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        templated = [_apply_sft_prompt(tokenizer, p) for p in prompts]
        enc = tokenizer(templated, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
    )
    # out.shape == (B * num_return_sequences, prompt_len + gen_len)

    B = len(prompts)
    prompt_len = input_ids.shape[1]
    gen_only = out[:, prompt_len:]  # strip prompt tokens
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    # Reshape (B*K,) → (B, K).
    return [decoded[b * num_return_sequences : (b + 1) * num_return_sequences] for b in range(B)]


def sample_k_valid_responses(
    model,
    tokenizer,
    prompts: list[str],
    K: int,
    *,
    word_count_constraints: dict[str, int],
    temperature: float = 1.0,
    max_new_tokens: int = 128,
    oversample_factor: int = 2,
    max_tries_per_doc: int = 8,
    device: torch.device | None = None,
) -> list[list[QSample]]:
    """Return K valid ``QSample``\\s per prompt via rejection sampling.

    Args:
        model: SFT ``CausalLM`` (``q_φ``) in eval mode.
        tokenizer: SFT tokenizer with a chat template (Qwen3-family).
        prompts: One document-text ``x`` per entry.
        K: Target valid samples per prompt.
        word_count_constraints: ``{"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}``.
        temperature: Sampling temperature. 1.0 = default, matches the
            distribution ``q_φ`` represents (no tempering). **Lowering
            temperature invalidates the ELBO** — we'd be sampling from
            ``q^{1/T}``, not ``q``. Don't.
        max_new_tokens: Per-response generation cap.
        oversample_factor: How many extra draws to make per retry pass.
        max_tries_per_doc: Give up on a prompt after this many draw rounds.
        device: Defaults to the model's device.

    Returns:
        ``result[b]`` = list of exactly ``K`` valid ``QSample``\\s for prompt b.

    Raises:
        RuntimeError: if any prompt can't reach K valid samples. The error
            reports the validity rate so the caller can decide whether to
            widen ``max_tries_per_doc`` or retrain ``q_φ``.
    """
    if device is None:
        device = next(model.parameters()).device

    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0 for stochastic sampling, got {temperature}. "
            "T != 1.0 samples from q^{1/T}, which is NOT the q referenced by the "
            "ELBO — using T != 1.0 yields an invalid bound."
        )
    if temperature != 1.0:
        log.warning(
            "sample_k_valid_responses called with temperature=%g != 1.0. "
            "This samples from a tempered q, so the resulting bound is "
            "with respect to q^{1/T}, not q_φ. Only acceptable for debugging.",
            temperature,
        )

    B = len(prompts)
    collected: list[list[QSample]] = [[] for _ in range(B)]
    total_drawn = [0] * B
    total_valid = [0] * B

    # Start by drawing K*oversample_factor per prompt; re-draw for stragglers.
    for _try_idx in range(max_tries_per_doc):
        # Only re-sample for prompts that haven't reached K yet.
        pending_idx = [b for b in range(B) if len(collected[b]) < K]
        if not pending_idx:
            break

        pending_prompts = [prompts[b] for b in pending_idx]
        # How many to draw this round: enough to fill (K - collected[b]) * oversample
        # for the worst pending doc; same number for all pending docs (batched).
        worst_deficit = max(K - len(collected[b]) for b in pending_idx)
        draws_this_round = max(1, worst_deficit * oversample_factor)

        responses_per_prompt = _generate_raw_responses(
            model,
            tokenizer,
            pending_prompts,
            num_return_sequences=draws_this_round,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        for b, responses in zip(pending_idx, responses_per_prompt, strict=True):
            for response in responses:
                total_drawn[b] += 1
                parsed = parse_layers(response, word_count_constraints, strict=True)
                if parsed is None:
                    continue
                total_valid[b] += 1
                if len(parsed) != len(LayerNames):
                    # parse_layers returns the z_4..z_1 order when it succeeds; we
                    # never expect this branch, but guard against a silent rewrite.
                    raise RuntimeError(
                        f"parse_layers returned {len(parsed)} layers; expected "
                        f"{len(LayerNames)}. Response: {response!r}"
                    )
                collected[b].append(QSample(response=response, layers=parsed))
                if len(collected[b]) == K:
                    break

    failed = [b for b in range(B) if len(collected[b]) < K]
    if failed:
        rates = [total_valid[b] / max(total_drawn[b], 1) for b in failed]
        raise RuntimeError(
            f"{len(failed)}/{B} prompts failed to reach K={K} valid samples "
            f"within {max_tries_per_doc} rounds. Min validity rate observed: "
            f"{min(rates):.2%}, median: {sorted(rates)[len(rates) // 2]:.2%}. "
            "Either raise max_tries_per_doc, reduce K, or check q_φ quality "
            "(the SFT model should be producing strictly-verified output)."
        )

    overall_rate = sum(total_valid) / max(sum(total_drawn), 1)
    log.info(
        "sample_k_valid_responses: %d prompts x K=%d, validity rate %.2f%% "
        "(drew %d total, %d valid).",
        B,
        K,
        100.0 * overall_rate,
        sum(total_drawn),
        sum(total_valid),
    )
    if overall_rate < 0.90:
        log.warning(
            "Validity rate %.2f%% is below 90%%. The rejection-sampling correction "
            "term -log q(valid|x) is no longer negligible; treat the IWAE-K bound "
            "as optimistic by up to %.3f nats per doc until you estimate it.",
            100.0 * overall_rate,
            -1.0 * float(torch.log(torch.tensor(overall_rate, dtype=torch.float64))),
        )
    return collected


__all__: list[Any] = ["QSample", "sample_k_valid_responses"]
