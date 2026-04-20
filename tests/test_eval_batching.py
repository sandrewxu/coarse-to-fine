"""Correctness tests for the parallelized NLL evaluators in ``src/eval``.

These replicate the core batching math from ``eval_ar`` / ``eval_diffusion``
in-place on a tiny Qwen3 model and verify it matches the unbatched reference
— there are no checkpoints available in the test env, so full ``eval_*``
drivers aren't exercised end-to-end.

Invariants under test:

1. **AR right-padding is equivalent to per-doc scoring.** Packing documents
   of mixed length with ``attention_mask`` + ``-100`` on pad targets must
   give each document the exact same NLL and token count it would get
   scored one at a time.

2. **Diffusion MC tiling is equivalent to sequential MC.** ``repeat_interleave``
   + ``view(B, m).sum(dim=1)`` must reduce to the same per-doc accumulator
   that the sequential inner loop produced, when both see the same ``t`` and
   mask draws.
"""

import math

import pytest
import torch

pytest.importorskip("transformers")


def _ar_batched_per_doc_nll(model, seqs, pad_id):
    """Replicate eval_ar's batched NLL path."""
    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    device = next(model.parameters()).device
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        input_ids[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[b, : len(s)] = 1

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:].clone()
    target_mask = attn[:, 1:].bool()
    targets = targets.masked_fill(~target_mask, -100)
    _, T, V = logits.shape
    per_tok = loss_fn(logits.reshape(-1, V), targets.reshape(-1)).view(B, T)
    nll = per_tok.sum(dim=1).tolist()
    ntok = target_mask.sum(dim=1).tolist()
    return nll, ntok


def _ar_unbatched_per_doc_nll(model, seqs):
    """Reference: the original eval_ar, one doc at a time."""
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    device = next(model.parameters()).device
    nlls, ntoks = [], []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor(s, dtype=torch.long, device=device).unsqueeze(0)
            out = model(input_ids=ids)
            logits = out.logits[:, :-1, :]
            targets = ids[:, 1:]
            per_tok = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            nlls.append(per_tok.sum().item())
            ntoks.append(int(targets.numel()))
    return nlls, ntoks


def _tiny_qwen3(vocab_size=32):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def test_ar_batched_matches_unbatched():
    """Per-doc NLL must be identical (fp-tolerance) whether docs are batched or not."""
    pad_id = 0
    bos_id = 2
    model = _tiny_qwen3(vocab_size=32)

    # Deliberately varied lengths so padding is exercised.
    docs = [
        [bos_id, 5, 6, 7, 8],
        [bos_id, 9, 10],
        [bos_id, 11, 12, 13, 14, 15, 16],
        [bos_id, 17, 18, 19],
    ]

    nll_b, ntok_b = _ar_batched_per_doc_nll(model, docs, pad_id=pad_id)
    nll_u, ntok_u = _ar_unbatched_per_doc_nll(model, docs)

    assert ntok_b == ntok_u
    for b, u in zip(nll_b, nll_u, strict=True):
        assert math.isclose(b, u, rel_tol=1e-5, abs_tol=1e-5), f"batched={b} vs unbatched={u}"


def test_ar_batch_of_one_matches_singleton():
    """Batched scoring with B=1 must match calling the model directly on that doc."""
    pad_id = 0
    bos_id = 2
    model = _tiny_qwen3(vocab_size=32)

    doc = [bos_id, 5, 6, 7, 8, 9]
    nll_b, ntok_b = _ar_batched_per_doc_nll(model, [doc], pad_id=pad_id)
    nll_u, ntok_u = _ar_unbatched_per_doc_nll(model, [doc])
    assert ntok_b == ntok_u
    assert math.isclose(nll_b[0], nll_u[0], rel_tol=1e-6, abs_tol=1e-6)


def test_diffusion_mc_tiling_matches_sequential():
    """Tiling docs along batch dim must sum to the same per-doc NELBO
    as calling ``mdlm_loss`` sequentially with matched random state.

    Approach: we seed torch identically, call ``mdlm_loss`` on a tiled
    ``(B*m, S)`` batch, and compare against ``m`` sequential calls on the
    ``(B, S)`` input where we pre-draw the random samples by hand. Because
    ``sample_t`` and ``q_xt`` consume randomness in a shape-dependent way,
    we factor the math directly rather than trying to rebind the RNG.
    """
    from src.c2f_model.training.diffusion import (
        LogLinearNoise,
        q_xt,
        subs_parameterization,
    )

    torch.manual_seed(123)
    mask_id = 31
    pad_id = 0
    bos_id = 2
    B, S, m = 2, 6, 4
    schedule = LogLinearNoise(eps=1e-3)
    model = _tiny_qwen3(vocab_size=32)

    x0 = torch.tensor(
        [
            [bos_id, 5, 6, 7, 8, 9],
            [bos_id, 10, 11, 12, 13, pad_id],
        ]
    )
    loss_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )

    # Pre-draw t and move_chance for each of the m replicates per doc.
    t = torch.rand(B * m) * 0.9 + 0.05
    sigma = schedule.total_noise(t)
    dsigma = schedule.rate_noise(t)
    move_chance = (1.0 - torch.exp(-sigma)).unsqueeze(-1)  # (B*m, 1)
    weight = (dsigma / torch.expm1(sigma)).unsqueeze(-1)  # (B*m, 1)

    # --- Path A: tiled, single forward ---
    torch.manual_seed(7)
    x0_rep = x0.repeat_interleave(m, dim=0)
    mask_rep = loss_mask.repeat_interleave(m, dim=0)
    xt_tiled = q_xt(x0_rep, move_chance, mask_id)
    with torch.no_grad():
        out_tiled = model(input_ids=xt_tiled)
    log_p_tiled = subs_parameterization(out_tiled.logits.float(), xt_tiled, mask_id)
    log_px0_tiled = log_p_tiled.gather(-1, x0_rep.unsqueeze(-1)).squeeze(-1)
    weighted_tiled = -log_px0_tiled * weight * mask_rep  # (B*m, S)
    per_rep_tiled = weighted_tiled.sum(dim=-1)  # (B*m,)
    doc_tiled = per_rep_tiled.view(B, m).sum(dim=1)  # (B,)

    # --- Path B: sequential m calls of B, same pre-drawn t and xt ---
    # Rebuild xt for each replicate using the same move_chance slice so the
    # mask pattern matches Path A exactly.
    torch.manual_seed(7)
    # Consume the same rand budget q_xt did in Path A (B*m*S draws).
    _ = torch.rand(B * m, S)
    doc_seq = torch.zeros(B)
    for rep in range(m):
        # The r'th replicate of doc b is row (b*m + r) in the tiled batch.
        idx = [b * m + rep for b in range(B)]
        xt_r = xt_tiled[idx]
        mc_r = move_chance[idx]  # noqa: F841 — shape-check only
        w_r = weight[idx]
        with torch.no_grad():
            out_r = model(input_ids=xt_r)
        log_p_r = subs_parameterization(out_r.logits.float(), xt_r, mask_id)
        log_px0_r = log_p_r.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        wnll_r = -log_px0_r * w_r * loss_mask
        doc_seq += wnll_r.sum(dim=-1)

    assert torch.allclose(doc_tiled, doc_seq, atol=1e-5, rtol=1e-5), (
        f"tiled={doc_tiled.tolist()} vs sequential={doc_seq.tolist()}"
    )
