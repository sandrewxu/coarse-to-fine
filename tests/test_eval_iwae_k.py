"""IWAE-K correctness for the C2F ELBO evaluator.

Covers three invariants:

1. **K=1 equivalence.** At K=1 the logsumexp aggregator must produce exactly
   the same per-doc bound as the existing gold-z combination, modulo float
   noise. This is a consistency check between the two code paths.

2. **Monotonicity in K.** IWAE-K is non-increasing in K as an upper bound on
   ``-log p(x)`` (Burda, Grosse, Salakhutdinov 2016). On synthetic log-weights
   drawn from a fixed distribution the empirical bound should respect this
   *in expectation*; we test that the estimated bound at K=4 is smaller than
   at K=1 on a seed-fixed draw with enough MC noise averaged out.

3. **Sampler rejects invalid outputs.** A stub model that produces responses
   of arbitrary length is run through ``sample_k_valid_responses``; the
   sampler must return only responses parsed by ``parse_layers``. Errors
   loudly when validity rate is too low to reach K.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")


def test_iwae_logsumexp_bound_at_k1_matches_plain_difference():
    """At K=1 the new aggregator must produce the K=1 gold-z math bit-for-bit."""
    from src.eval.bound import _iwae_logsumexp_bound

    rng = np.random.default_rng(0)
    joint_nll_p = rng.uniform(50.0, 150.0, size=(7, 1))
    nll_q = rng.uniform(10.0, 40.0, size=(7, 1))

    bound = _iwae_logsumexp_bound(joint_nll_p, nll_q, K=1)
    expected = (joint_nll_p - nll_q).squeeze(-1)

    np.testing.assert_allclose(bound, expected, rtol=1e-12, atol=1e-12)


def test_iwae_logsumexp_bound_monotone_in_k_on_synthetic_draws():
    """Bound at K=16 should be <= bound at K=1 on the same log-weight draws.

    Draw M samples from a fixed (joint, q) distribution per doc, aggregate with
    K=1 and K=16 on a common subset. E_{z_1:K}[bound_K] is monotone in K; with
    enough docs + a deterministic truncation of draws to (K=1, K=16), the
    empirical per-doc means should also be ordered for a wide enough gap.
    """
    from src.eval.bound import _iwae_logsumexp_bound

    rng = np.random.default_rng(42)
    B, M = 200, 16
    # Synthetic: log weights roughly N(-30, 3^2) — realistic scale for short
    # sequences where joint_nll_p ~ 150 nats and nll_q ~ 120 nats.
    log_w_all = rng.normal(loc=-30.0, scale=3.0, size=(B, M))
    # Reconstruct (joint_nll_p, nll_q) s.t. nll_q - joint_nll_p == log_w:
    # pick joint_nll_p ~ Unif(50, 150), then nll_q = joint_nll_p + log_w.
    joint_nll_p = rng.uniform(50.0, 150.0, size=(B, M))
    nll_q = joint_nll_p + log_w_all

    # K=1: use only the first draw per doc.
    bound_k1 = _iwae_logsumexp_bound(joint_nll_p[:, :1], nll_q[:, :1], K=1)
    bound_k16 = _iwae_logsumexp_bound(joint_nll_p, nll_q, K=M)

    # Each per-doc pair: K=M bound is logsumexp average, K=1 is a single pick.
    # Jensen: E[K=1 bound] >= E[K=M bound]. Empirically on 200 docs this holds
    # strictly in the mean; we test the stronger per-doc claim only in aggregate.
    assert bound_k16.mean() < bound_k1.mean(), (
        f"K=16 bound (mean={bound_k16.mean():.3f}) not tighter than "
        f"K=1 bound (mean={bound_k1.mean():.3f})."
    )


def test_iwae_logsumexp_bound_matches_hand_computed_small_case():
    """Exact numeric match against a hand-computed K=3 case.

    K=3, one doc, log_w = [-2.0, -1.0, -0.5]:
        logsumexp = log(e^-2 + e^-1 + e^-0.5) = log(0.1353 + 0.3679 + 0.6065)
                  = log(1.1098) ≈ 0.1042
        bound     = log(3) - logsumexp = 1.0986 - 0.1042 ≈ 0.9944
    """
    from src.eval.bound import _iwae_logsumexp_bound

    joint_nll_p = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    nll_q = np.array([[-2.0, -1.0, -0.5]], dtype=np.float64)
    # log_w = nll_q - joint_nll_p = [-2.0, -1.0, -0.5]

    bound = _iwae_logsumexp_bound(joint_nll_p, nll_q, K=3)
    expected = np.log(3.0) - np.log(np.exp(-2.0) + np.exp(-1.0) + np.exp(-0.5))
    np.testing.assert_allclose(bound, np.asarray([expected]), rtol=1e-10)


# --------------------------------------------------------------------------
# Sampler tests — uses a stub tokenizer + hand-written responses to exercise
# the validity gate without actually running HF generate.
# --------------------------------------------------------------------------


class _StubModel:
    """Minimal model stub: generate() returns a pre-baked id tensor."""

    def __init__(self, response_strings: list[str], tokenizer) -> None:
        self._responses = response_strings
        self._tok = tokenizer
        self.device = torch.device("cpu")

    def parameters(self):
        # sample_k_valid_responses does next(model.parameters()).device.
        yield torch.zeros(1, device=self.device)

    @torch.no_grad()
    def generate(self, *, input_ids, num_return_sequences, **kwargs):
        B = input_ids.shape[0]

        # Encode each pre-baked response; pad them all to a common length.
        encoded: list[list[int]] = []
        for _ in range(B * num_return_sequences):
            # Cycle through the response list deterministically.
            idx = len(encoded) % len(self._responses)
            text = self._responses[idx]
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            encoded.append(ids)
        max_gen = max(len(ids) for ids in encoded)
        pad_id = self._tok.pad_token_id or self._tok.eos_token_id
        gen_pad: list[list[int]] = []
        for ids in encoded:
            ids = ids + [pad_id] * (max_gen - len(ids))
            gen_pad.append(ids)
        gen_tensor = torch.tensor(gen_pad, dtype=torch.long)  # (B*K, max_gen)

        # Concatenate prompt_ids + generated_ids along seq dim.
        prompt_rep = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([prompt_rep, gen_tensor], dim=1)


def _tiny_tokenizer():
    """Reuse the test_eval_bound helper."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
    except Exception:
        pytest.skip("no HF network / cache for a chat-template tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_sampler_rejects_invalid_and_returns_k_valid():
    """Stub model emits one valid response and one invalid per prompt.

    The sampler must keep retrying (via rejection + over-sampling) until it
    gathers K valid samples per prompt.
    """
    from src.eval.q_sampler import sample_k_valid_responses

    tok = _tiny_tokenizer()
    word_count_constraints = {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}

    # Valid: exact word counts.
    valid_response = (
        "z_4: aa bb\n"
        "z_3: aa bb cc dd\n"
        "z_2: aa bb cc dd ee ff gg hh\n"
        "z_1: aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp"
    )
    # Invalid: truncated — z_1 is missing words.
    invalid_response = "z_4: aa bb\nz_3: aa bb cc dd\nz_2: aa bb cc dd ee ff gg hh\nz_1: aa bb"

    model = _StubModel([valid_response, invalid_response], tok)
    prompts = ["first prompt", "second prompt"]

    out = sample_k_valid_responses(
        model,
        tok,
        prompts,
        K=2,
        word_count_constraints=word_count_constraints,
        temperature=1.0,
        max_new_tokens=64,
        oversample_factor=2,
        max_tries_per_doc=4,
    )
    assert len(out) == 2
    for per_prompt in out:
        assert len(per_prompt) == 2, f"expected K=2 valid samples, got {len(per_prompt)}"
        for qs in per_prompt:
            assert qs.response == valid_response, (
                f"sampler returned an invalid-looking response: {qs.response!r}"
            )
            assert len(qs.layers) == 4
            assert len(qs.layers[0].split()) == 2  # z_4


def test_sampler_errors_loudly_when_validity_rate_is_zero():
    """Every sample invalid → should raise rather than loop forever or silently truncate."""
    from src.eval.q_sampler import sample_k_valid_responses

    tok = _tiny_tokenizer()
    word_count_constraints = {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}

    invalid_only = ["malformed garbage not a z_ line at all."]
    model = _StubModel(invalid_only, tok)

    with pytest.raises(RuntimeError, match="failed to reach K="):
        sample_k_valid_responses(
            model,
            tok,
            ["prompt"],
            K=2,
            word_count_constraints=word_count_constraints,
            temperature=1.0,
            max_new_tokens=16,
            oversample_factor=2,
            max_tries_per_doc=3,
        )


def test_sampler_rejects_temperature_zero():
    """T=0 would collapse to argmax, invalidating the stochastic proposal q."""
    from src.eval.q_sampler import sample_k_valid_responses

    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="temperature must be > 0"):
        sample_k_valid_responses(
            _StubModel(["x"], tok),
            tok,
            ["p"],
            K=1,
            word_count_constraints={"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16},
            temperature=0.0,
        )
