"""MDLM (continuous-time SUBS) NELBO evaluator.

Computes a Monte-Carlo lower bound on ``log p(x)`` for the masked-diffusion
baseline trained by ``scripts/06c_train_diffusion_baseline.py``. The math is
shared bit-for-bit with the training loss in
``src/c2f_model/training/diffusion.py`` so training and eval cannot drift.

For each test document, we draw ``N`` samples of ``(t, mask)`` and average the
per-position weighted CE — that's the NELBO estimate. Default ``N=128``
honors the ``08`` stub's planned tightness; raise ``N`` if MC noise dominates
the doc-level variance in the bootstrap CI.

MC samples are drawn iid and independent across docs, so we batch them along
the batch dim: each doc is tiled ``mc_batch_size`` times, fed through the
model in one forward pass, and the per-doc NELBOs averaged. This is
statistically equivalent to the sequential inner loop (same distribution of
``t`` and ``mask`` per replicate) while cutting forward-pass count by that
factor.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.logging import get_logger
from src.eval.common import (
    bootstrap_ci,
    check_vocab_consistency,
    load_space_tokenizer,
    load_test_docs,
)

log = get_logger(__name__)


def _mc_nelbo_batch(
    model,
    x0: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_id: int,
    pad_id: int,
    schedule,
    eps_t: float,
    N: int,
    mc_batch_size: int,
) -> torch.Tensor:
    """MC-NELBO per doc for one doc batch, packing ``mc_batch_size`` iid samples per forward.

    MC draws are iid and independent across docs, so tiling along the batch
    dim is statistically equivalent to the sequential inner loop.
    """
    from src.c2f_model.training.diffusion import mdlm_loss

    B = x0.shape[0]
    doc_accum = torch.zeros(B, dtype=torch.float64, device=x0.device)
    remaining = N
    while remaining > 0:
        m = min(mc_batch_size, remaining)
        x0_rep = x0.repeat_interleave(m, dim=0)
        mask_rep = loss_mask.repeat_interleave(m, dim=0)
        sample_nll = mdlm_loss(
            model,
            x0_rep,
            mask_rep,
            mask_id=mask_id,
            pad_id=pad_id,
            schedule=schedule,
            eps_t=eps_t,
            # iid draws across MC samples — antithetic would correlate within
            # the batch axis, which is wrong for independent MC replicates.
            antithetic=False,
            reduction="per_doc",
        )  # (B*m,)
        doc_accum += sample_nll.view(B, m).sum(dim=1).to(torch.float64)
        remaining -= m
    return doc_accum / N


@torch.no_grad()
def eval_diffusion(
    *,
    ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None = None,
    tokenizer_dir: Path | None = None,
    N: int = 128,
    batch_size: int = 8,
    mc_batch_size: int = 16,
) -> dict[str, Any]:
    """Compute MC-NELBO for a SUBS-MDLM checkpoint on a JSONL test set.

    Args:
        ckpt: Diffusion checkpoint dir (saved by step 6c).
        test: JSONL test file (same format as the AR evaluator).
        config: Loaded experiment config dict.
        limit: Cap number of docs scored.
        tokenizer_dir: Override the space tokenizer dir (else from config).
        N: MC samples per document.
        batch_size: Documents per forward pass.
        mc_batch_size: MC samples packed in parallel per forward. Effective
            batch is ``batch_size * mc_batch_size`` — lower if OOM, raise to
            use more of the GPU. Clamped to ``N``.

    Returns:
        Dict matching AR/C2F evaluators with extra ``N`` field.
    """
    from transformers import AutoModelForCausalLM

    from src.c2f_model.training.diffusion import LogLinearNoise

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.c2f_model.training.tokenizer import MASK_TOKEN

    tokenizer = load_space_tokenizer(config, tokenizer_dir)
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    pad_id = tokenizer.pad_token_id
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Tokenizer does not contain {MASK_TOKEN!r} as a distinct id; "
            "retrain the tokenizer to match the diffusion training setup."
        )

    log.info("Loading diffusion model from %s...", ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), trust_remote_code=True, attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    check_vocab_consistency(model.config.vocab_size, tokenizer.vocab_size)

    df_cfg = config.get("diffusion_training", {})
    schedule = LogLinearNoise(eps=df_cfg.get("noise_eps", 1e-3))
    eps_t = df_cfg.get("eps_t", 1e-3)
    text_word_count = config.get("text_word_count", 32)

    docs = load_test_docs(test, limit, text_word_count=text_word_count)
    mc_batch_size = max(1, min(mc_batch_size, N))
    log.info(
        "Scoring %d documents with N=%d MC samples (mc_batch_size=%d)...",
        len(docs),
        N,
        mc_batch_size,
    )

    bos = tokenizer.bos_token_id or tokenizer.eos_token_id

    # Pre-tokenize all docs to fixed-length [BOS, w_0, ..., w_{T-1}].
    seq_len = 1 + text_word_count
    all_ids = torch.empty((len(docs), seq_len), dtype=torch.long)
    per_doc_tokens = np.empty(len(docs), dtype=np.int64)
    for i, text in enumerate(docs):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= text_word_count:
            ids = ids[:text_word_count]
            n_tok = text_word_count
        else:
            n_tok = len(ids)
            ids = ids + [pad_id] * (text_word_count - n_tok)
        all_ids[i] = torch.tensor([bos, *ids], dtype=torch.long)
        per_doc_tokens[i] = n_tok  # nats/word denominator: real text tokens only

    per_doc_nll = np.zeros(len(docs), dtype=np.float64)
    rows: list[dict] = []

    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        B = end - start
        x0 = all_ids[start:end].to(device)  # (B, S)
        # loss_mask: 1 at content tokens (positions 1..n_tok), 0 at BOS / padding.
        loss_mask = torch.zeros_like(x0, dtype=torch.float32)
        for b in range(B):
            n_tok = int(per_doc_tokens[start + b])
            loss_mask[b, 1 : 1 + n_tok] = 1.0

        doc_accum = _mc_nelbo_batch(
            model,
            x0,
            loss_mask,
            mask_id=mask_id,
            pad_id=pad_id,
            schedule=schedule,
            eps_t=eps_t,
            N=N,
            mc_batch_size=mc_batch_size,
        )

        for b in range(B):
            doc_idx = start + b
            nll = float(doc_accum[b].item())
            n_tok = int(per_doc_tokens[doc_idx])
            per_doc_nll[doc_idx] = nll
            rows.append({"idx": doc_idx, "nll": nll, "num_tokens": n_tok})

    point, lo, hi = bootstrap_ci(per_doc_nll, per_doc_tokens)
    return {
        "model_kind": "diffusion",
        "ckpt": str(ckpt),
        "num_docs": len(docs),
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "N": N,
        "per_doc_rows": rows,
    }
