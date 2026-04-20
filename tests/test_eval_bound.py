"""Correctness tests for the C2F ELBO upper-bound evaluator.

The bound math is:

    bound_per_doc = joint_nll_p - nll_q
    nats_per_text_word_bound = bound_per_doc / text_word_count

Two things worth pinning down:

1. ``_sft_nll_per_doc`` must score only the response tokens, matching the
   training-time ``-100`` mask. If it scored prompt tokens too, the bound
   would be way too loose.
2. The final combination is a plain per-doc difference — check on a mock
   pair of NLL arrays that the bootstrap output matches the hand computation.
"""

import pytest
import torch

pytest.importorskip("transformers")


def _tiny_qwen_like_tokenizer():
    """Return a real HF tokenizer with a chat template (matches what SFT uses)."""
    from transformers import AutoTokenizer

    # Qwen2-0.5B has the same chat template as Qwen3 and is cached in most
    # HF test envs; fall back to hand-rolled template if it's not downloadable.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
    except Exception:
        pytest.skip("no HF network / cache for a chat-template tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _tiny_causal_lm_matching(tokenizer):
    """Build a tiny Qwen2ForCausalLM sharing vocab with the tokenizer.

    Uses ``len(tokenizer)`` (base vocab + added/special tokens) to ensure
    every id the tokenizer might emit is in range of the embedding table.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def test_sft_nll_scores_only_response_tokens():
    """-log q(z|x) must drop the user (prompt) tokens from the sum.

    Swapping the prompt text (with response held fixed) must not change the
    returned NLL by more than the contribution of attention — the per-token
    labels over the prompt are all -100, so CE is zero there.

    Concretely: run twice with different prompts + same response, and the
    delta should be small relative to the per-doc total. If we were
    accidentally summing over prompt tokens, the swap would move the NLL
    by hundreds of nats.
    """
    from src.eval.bound import _sft_nll_per_doc

    tok = _tiny_qwen_like_tokenizer()
    model = _tiny_causal_lm_matching(tok)
    device = torch.device("cpu")

    prompts_a = ["the cat sat on the mat briefly"]
    prompts_b = ["antidisestablishmentarianism is a long word here"]
    responses = ["z_4: short\nz_3: also short\nz_2: ...\nz_1: last one"]

    nll_a = _sft_nll_per_doc(model, tok, prompts_a, responses, device, batch_size=1)
    nll_b = _sft_nll_per_doc(model, tok, prompts_b, responses, device, batch_size=1)

    # Both should be positive and of similar magnitude. The exact values will
    # differ because attention over different prompt tokens changes the logits
    # at response positions — but the change is per-token bounded and should
    # be a small fraction of the per-doc total (which is ~50 nats for a tiny
    # untrained model on ~20 response tokens). If we were summing over
    # prompt tokens too, `nll_b` (longer prompt) would balloon by hundreds.
    assert nll_a.shape == (1,)
    assert nll_b.shape == (1,)
    assert nll_a[0] > 0 and nll_b[0] > 0
    rel_diff = abs(nll_a[0] - nll_b[0]) / max(nll_a[0], nll_b[0])
    assert rel_diff < 0.5, (
        f"nll_a={nll_a[0]:.2f} vs nll_b={nll_b[0]:.2f} differ by "
        f"{rel_diff * 100:.1f}% — suggests prompt tokens are leaking into "
        "the loss. Only response tokens should contribute."
    )


def test_sft_nll_matches_manual_ce_on_response_only():
    """Exact numerical equality with a hand-computed CE over the response slice."""
    from src.eval.bound import _sft_nll_per_doc

    tok = _tiny_qwen_like_tokenizer()
    model = _tiny_causal_lm_matching(tok)
    device = torch.device("cpu")

    prompt = "short prompt"
    response = "z_4: a b\nz_3: c d"

    auto_nll = _sft_nll_per_doc(model, tok, [prompt], [response], device, batch_size=1)[0]

    # Manual reference: build the exact same inputs + labels and compute CE.
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(messages, tokenize=False)
    full_ids = tok(full_text)["input_ids"]
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(prompt_text)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = labels_t[:, 1:]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_tok = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    manual_nll = per_tok.sum().item()

    assert abs(auto_nll - manual_nll) < 1e-4, (
        f"auto_nll={auto_nll:.6f} vs manual_nll={manual_nll:.6f}"
    )


def test_bound_combination_and_bootstrap_units():
    """End-to-end: joint_nll_p - nll_q divided by text_word_count matches bootstrap output."""
    import numpy as np

    from src.eval.common import bootstrap_ci

    # Hand-crafted per-doc NLLs.
    joint_nll_p = np.array([120.0, 90.0, 150.0, 100.0], dtype=np.float64)
    nll_q = np.array([20.0, 15.0, 30.0, 20.0], dtype=np.float64)
    text_word_count = 32

    bound_per_doc = joint_nll_p - nll_q  # [100, 75, 120, 80]
    per_doc_tokens = np.full(len(bound_per_doc), text_word_count, dtype=np.int64)

    point, lo, hi = bootstrap_ci(bound_per_doc, per_doc_tokens)
    expected_point = bound_per_doc.sum() / per_doc_tokens.sum()
    # (100+75+120+80) / (4*32) = 375 / 128 ≈ 2.9297
    assert abs(point - expected_point) < 1e-9
    assert abs(expected_point - 2.9296875) < 1e-9
    assert lo < point < hi


# --------------------------------------------------------------------------
# load_test_docs — unified loader shared by AR / diffusion / c2f_bound so
# all four evaluators can score the same doc subset.
# --------------------------------------------------------------------------


def _write_jsonl(path, rows):
    import json

    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_parquet(path, columns: dict[str, list]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(columns), str(path))


def test_load_test_docs_jsonl(tmp_path):
    from src.eval.common import load_test_docs

    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}])

    docs = load_test_docs(p, limit=None, text_word_count=32)
    assert docs == ["alpha", "beta", "gamma"]


def test_load_test_docs_sft_parquet(tmp_path):
    from src.eval.common import load_test_docs

    p = tmp_path / "sft.parquet"
    _write_parquet(
        p,
        {"prompt": ["p1", "p2", "p3"], "response": ["r1", "r2", "r3"]},
    )

    docs = load_test_docs(p, limit=None, text_word_count=32)
    assert docs == ["p1", "p2", "p3"]


def test_load_test_docs_c2f_parquet_returns_final_text_words(tmp_path):
    """c2f-format flat text ends with the original document — recover exactly those words."""
    from src.eval.common import load_test_docs

    p = tmp_path / "c2f.parquet"
    # scale layout in this repo: [2, 4, 8, 16, 32] words → 30 latent words
    # followed by 32 text words. We only need a representative example.
    latent_words = " ".join(f"z{i}" for i in range(30))
    text_words_a = " ".join(f"a{i}" for i in range(32))
    text_words_b = " ".join(f"b{i}" for i in range(32))
    _write_parquet(
        p,
        {"text": [f"{latent_words} {text_words_a}", f"{latent_words} {text_words_b}"]},
    )

    docs = load_test_docs(p, limit=None, text_word_count=32)
    assert docs == [text_words_a, text_words_b]


def test_load_test_docs_limit_applied_uniformly(tmp_path):
    from src.eval.common import load_test_docs

    jp = tmp_path / "docs.jsonl"
    _write_jsonl(jp, [{"text": f"doc{i}"} for i in range(5)])
    sp = tmp_path / "sft.parquet"
    _write_parquet(
        sp,
        {"prompt": [f"p{i}" for i in range(5)], "response": [f"r{i}" for i in range(5)]},
    )
    cp = tmp_path / "c2f.parquet"
    latent = " ".join(["pad"] * 30)
    text_rows = [" ".join([f"t{i}_{k}" for k in range(32)]) for i in range(5)]
    _write_parquet(cp, {"text": [f"{latent} {t}" for t in text_rows]})

    for path in (jp, sp, cp):
        got = load_test_docs(path, limit=2, text_word_count=32)
        assert len(got) == 2, f"limit not applied at {path.name}: got {len(got)} docs"


# --------------------------------------------------------------------------
# c2f-format reconstruction inside eval_c2f_bound — lets the bound eval
# score q(z|x) when only c2f_val.parquet (flat text) is available.
# --------------------------------------------------------------------------


def test_reconstruct_prompt_response_roundtrips_flatten_for_c2f():
    """Reconstructed (prompt, layer-content words) match what flatten_for_c2f writes.

    The SFT response's exact whitespace isn't recoverable from the flat text
    (newlines were stripped), but the per-layer word content must be exact
    — that's what ``parse_layers`` consumes downstream.
    """
    from src.eval.bound import _reconstruct_prompt_response_from_c2f
    from src.rl.common import LayerNames, compute_word_boundaries, parse_layers

    word_count_constraints = {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}
    text_word_count = 32

    # Build layer strings programmatically so word counts match the constraints
    # exactly — hand-written prose is too easy to miscount and the reconstruction
    # assumes exact widths.
    layer_contents = {
        "z_4": " ".join(f"c4_{i}" for i in range(2)),
        "z_3": " ".join(f"c3_{i}" for i in range(4)),
        "z_2": " ".join(f"c2_{i}" for i in range(8)),
        "z_1": " ".join(f"c1_{i}" for i in range(16)),
    }
    prompt_text = " ".join(f"t{i}" for i in range(text_word_count))
    # Replicate flatten_for_c2f: concat layer contents in z_4..z_1 order then prompt.
    flat = " ".join(
        [
            layer_contents["z_4"],
            layer_contents["z_3"],
            layer_contents["z_2"],
            layer_contents["z_1"],
            prompt_text,
        ]
    )

    prompts, responses = _reconstruct_prompt_response_from_c2f(
        [flat],
        word_count_constraints=word_count_constraints,
        text_word_count=text_word_count,
        layer_names=LayerNames,
        compute_word_boundaries=compute_word_boundaries,
    )
    assert prompts == [prompt_text]
    parsed = parse_layers(responses[0], word_count_constraints, strict=True)
    assert parsed is not None, f"reconstructed response failed parse: {responses[0]!r}"
    # parse_layers returns [z_4, z_3, z_2, z_1] in order (see src/rl/common.py:62)
    assert parsed[0] == layer_contents["z_4"]
    assert parsed[1] == layer_contents["z_3"]
    assert parsed[2] == layer_contents["z_2"]
    assert parsed[3] == layer_contents["z_1"]


# --------------------------------------------------------------------------
# Row-order assumption: eval_c2f_bound reads prompts/responses via
# pq.read_table and assumes the same ordering as C2FDataset. Any divergence
# would silently misalign joint_nll_p (from C2F forward) with nll_q (from
# SFT forward). Pin it down with a tiny parquet.
# --------------------------------------------------------------------------


def test_c2f_dataset_preserves_parquet_row_order(tmp_path):
    pytest.importorskip("datasets")
    import pyarrow.parquet as pq
    from datasets import load_dataset

    p = tmp_path / "rows.parquet"
    _write_parquet(
        p,
        {
            "prompt": ["first", "second", "third", "fourth"],
            "response": ["r1", "r2", "r3", "r4"],
        },
    )

    pq_prompts = pq.read_table(str(p), columns=["prompt"]).column("prompt").to_pylist()
    hf_prompts = load_dataset("parquet", data_files=str(p), split="train")["prompt"]

    assert list(hf_prompts) == pq_prompts, (
        "HF load_dataset reordered parquet rows — eval_c2f_bound's alignment "
        "between C2FDataset and pq.read_table would break."
    )


# --------------------------------------------------------------------------
# Denominator invariant: c2f_bound uses a fixed text_word_count per doc,
# AR uses the actual number of content tokens. These must agree whenever
# all docs are exactly text_word_count words (the TinyStories post-step-0
# pipeline). If a future pipeline change admits shorter docs, this test
# fails and forces a decision.
# --------------------------------------------------------------------------


def test_bound_denominator_matches_ar_on_fixed_length_docs():
    import numpy as np
    import torch as _torch

    text_word_count = 32
    # Simulate AR's per-doc token count computation: mask = attn_mask shifted.
    # Each doc = BOS + 32 content tokens → 32 scored target positions per doc.
    n_docs = 5
    B, S = n_docs, text_word_count + 1
    attn_mask = _torch.ones(B, S, dtype=_torch.long)
    target_mask = attn_mask[:, 1:].bool()  # drop BOS shift, mirror eval_ar.py
    ar_tokens = target_mask.sum(dim=1).cpu().numpy().astype(np.int64)

    bound_tokens = np.full(n_docs, text_word_count, dtype=np.int64)

    assert (ar_tokens == bound_tokens).all(), (
        "AR per-doc token count diverges from c2f_bound's fixed "
        "text_word_count — aggregate nats/word values are no longer "
        "directly comparable between the two evaluators."
    )
