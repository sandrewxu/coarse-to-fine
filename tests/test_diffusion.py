"""Tests for the MDLM masked-diffusion baseline.

The two load-bearing invariants are:

1. ``subs_parameterization`` enforces the carry-over-unmasking math:
   log-prob at MASK is ``-inf``, unmasked positions force their observed
   token to log-prob 0 (everything else ``-inf``), and the rows are valid
   log-probabilities (logsumexp == 0).

2. The 4D bidirectional attention mask actually gives bidirectional
   attention through Qwen3 (transformers ≥4.45 contract). Skipped when
   ``transformers`` is not installed in the local env (it's a GPU-only
   dep per ``CLAUDE.md``).
"""

import math

import pytest
import torch

# diffusion.py imports `transformers` (for Trainer) and is reached via
# src.c2f_model.__init__ which imports Qwen3Config — both unavailable in the
# CPU-only local env. Skip the entire module rather than fail collection.
pytest.importorskip("transformers")

from src.c2f_model.training.diffusion import (
    NEG_INF,
    LogLinearNoise,
    make_bidirectional_4d_mask,
    q_xt,
    sample_t,
    subs_parameterization,
)


def test_loglinear_schedule_endpoints():
    sched = LogLinearNoise(eps=1e-3)
    # σ(0) ≈ 0, σ(1) is large but finite.
    assert sched.total_noise(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-6)
    one = sched.total_noise(torch.tensor(1.0)).item()
    assert one > 5.0  # -log(eps) ≈ 6.9
    assert math.isfinite(one)
    # rate is positive and grows.
    r0 = sched.rate_noise(torch.tensor(0.0)).item()
    r1 = sched.rate_noise(torch.tensor(0.999)).item()
    assert 0 < r0 < r1


def test_loglinear_weight_simplifies_to_inv_t():
    """For LogLinear, dσ/(eᵟ-1) ≈ 1/t (up to ε correction)."""
    sched = LogLinearNoise(eps=1e-3)
    t = torch.tensor([0.1, 0.3, 0.5, 0.9])
    sigma = sched.total_noise(t)
    dsigma = sched.rate_noise(t)
    weight = dsigma / torch.expm1(sigma)
    # Weight should be ≈ 1/t (within a few percent given ε=1e-3).
    expected = 1.0 / t
    assert torch.allclose(weight, expected, rtol=2e-3)


def test_subs_parameterization_invariants():
    """SUBS: MASK gets -inf; unmasked positions force the observed token; rows are valid log-probs."""
    torch.manual_seed(0)
    B, S, V = 2, 4, 6
    mask_id = 5
    logits = torch.randn(B, S, V, dtype=torch.float32)
    # xt: positions 0 and 2 are masked, others are real tokens.
    xt = torch.tensor([[0, mask_id, 1, mask_id], [2, 3, mask_id, 4]])

    log_p = subs_parameterization(logits, xt, mask_id)

    # (a) MASK index column is -inf everywhere.
    assert torch.all(log_p[:, :, mask_id] == NEG_INF)

    # (b) Unmasked positions: log_p[i, j, xt[i,j]] == 0 and others are -inf.
    unmasked = xt != mask_id
    for b in range(B):
        for s in range(S):
            if unmasked[b, s]:
                obs = xt[b, s].item()
                row = log_p[b, s]
                assert row[obs].item() == 0.0
                others = torch.cat([row[:obs], row[obs + 1 :]])
                assert torch.all(others == NEG_INF)

    # (c) Rows for masked positions are valid log-probs (logsumexp == 0).
    for b in range(B):
        for s in range(S):
            if not unmasked[b, s]:
                row = log_p[b, s]
                # The MASK entry is -inf and the rest is a normalized log-softmax.
                lse = torch.logsumexp(row, dim=-1).item()
                assert lse == pytest.approx(0.0, abs=1e-5)


def test_q_xt_masks_proportionally():
    """At move_chance=0, no tokens are masked; at 1, all are."""
    torch.manual_seed(0)
    x0 = torch.arange(20).reshape(4, 5)
    mask_id = 99

    no_mask = q_xt(x0, torch.zeros(4, 1), mask_id)
    assert torch.equal(no_mask, x0)

    all_mask = q_xt(x0, torch.ones(4, 1), mask_id)
    assert torch.all(all_mask == mask_id)

    # Roughly the right rate at p=0.5 over many tokens.
    torch.manual_seed(0)
    x_big = torch.zeros(1000, 100, dtype=torch.long)
    half = q_xt(x_big, torch.full((1000, 1), 0.5), mask_id)
    rate = (half == mask_id).float().mean().item()
    assert 0.45 < rate < 0.55


def test_sample_t_in_bounds():
    eps_t = 1e-3
    t = sample_t(64, device=torch.device("cpu"), eps_t=eps_t, antithetic=False)
    assert torch.all(t >= eps_t) and torch.all(t <= 1.0)
    # Antithetic spreads samples more uniformly (mean ≈ 0.5).
    t_anti = sample_t(1024, device=torch.device("cpu"), eps_t=eps_t, antithetic=True)
    assert abs(t_anti.mean().item() - 0.5) < 0.02


def test_bidirectional_mask_shape_and_padding():
    """Padding positions should be -inf (key dim), real positions 0."""
    pad_id = 0
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])  # pad in trailing positions
    mask = make_bidirectional_4d_mask(input_ids, pad_id, dtype=torch.float32)
    assert mask.shape == (2, 1, 4, 4)
    # All real key positions are 0 (no penalty); pad keys are finfo.min.
    fmin = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 0].item() == 0.0  # row 0: key 0 is real
    assert mask[0, 0, 0, 2].item() == fmin  # row 0: key 2 is pad
    assert mask[1, 0, 0, 3].item() == fmin  # row 1: key 3 is pad
    assert mask[1, 0, 0, 2].item() == 0.0  # row 1: key 2 is real


def test_qwen3_with_4d_mask_is_bidirectional():
    """Load-bearing invariant: a 4D zero mask must give bidirectional attention.

    Concretely: changing a token at position j>i must change the logits at
    position i (which would NOT happen under causal masking). Skipped locally
    where ``transformers`` isn't installed (it's a GPU-only dep).
    """
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=8,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).eval()

    x_a = torch.tensor([[10, 20, 30, 40]])
    x_b = torch.tensor([[10, 20, 30, 41]])  # only position 3 differs
    mask = make_bidirectional_4d_mask(x_a, pad_id=0, dtype=torch.float32)

    with torch.no_grad():
        log_a = model(input_ids=x_a, attention_mask=mask).logits
        log_b = model(input_ids=x_b, attention_mask=mask).logits

    # If attention were causal, position 0's logits would be identical (it
    # cannot see positions 1-3). Bidirectional => they MUST differ.
    diff = (log_a[0, 0] - log_b[0, 0]).abs().max().item()
    assert diff > 1e-5, "Position 0 logits unchanged → attention is still causal"
