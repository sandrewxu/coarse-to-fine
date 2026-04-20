"""MDLM (continuous-time SUBS) NELBO evaluator.

Computes a Monte-Carlo lower bound on ``log p(x)`` for the masked-diffusion
baseline trained by ``scripts/06c_train_diffusion_baseline.py``. The math is
shared bit-for-bit with the training loss in
``src/c2f_model/training/diffusion.py`` so training and eval cannot drift.

For each test document, we draw ``N`` samples of ``(t, mask)`` and average the
per-position weighted CE — that's the NELBO estimate. Default ``N=128``
honors the ``08`` stub's planned tightness; raise ``N`` if MC noise dominates
the doc-level variance in the bootstrap CI.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.logging import get_logger
from src.eval.common import bootstrap_ci, check_vocab_consistency, load_space_tokenizer

log = get_logger(__name__)


def _load_jsonl(path: Path, limit: int | None) -> list[str]:
    """Same loader as ``src.eval.ar`` — keeps the AR/diffusion test paths identical."""
    docs: list[str] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append(obj.get("text", obj) if isinstance(obj, dict) else obj)
            except json.JSONDecodeError:
                docs.append(line)
    return docs


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

    Returns:
        Dict matching AR/C2F evaluators with extra ``N`` field.
    """
    from transformers import AutoModelForCausalLM

    from src.c2f_model.training.diffusion import LogLinearNoise, mdlm_loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = load_space_tokenizer(config, tokenizer_dir)
    mask_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id
    if mask_id is None:
        raise RuntimeError(
            "Tokenizer has no unk_token_id; MDLM eval requires [UNK] as the MASK index."
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

    docs = _load_jsonl(test, limit)
    log.info("Scoring %d documents with N=%d MC samples...", len(docs), N)

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

    # MC samples per doc, batched across docs for GPU efficiency.
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        x0 = all_ids[start:end].to(device)  # (B, S)
        # loss_mask: 1 at content tokens (positions 1..n_tok), 0 at BOS / padding.
        loss_mask = torch.zeros_like(x0, dtype=torch.float32)
        for b in range(end - start):
            n_tok = int(per_doc_tokens[start + b])
            loss_mask[b, 1 : 1 + n_tok] = 1.0

        doc_accum = torch.zeros(end - start, dtype=torch.float64, device=device)
        for _ in range(N):
            sample_nll = mdlm_loss(
                model,
                x0,
                loss_mask,
                mask_id=mask_id,
                pad_id=pad_id,
                schedule=schedule,
                eps_t=eps_t,
                # Antithetic sampling within a fixed batch is meaningless here
                # (we want independent draws across MC samples), so disable.
                antithetic=False,
                reduction="per_doc",
            )
            doc_accum += sample_nll.to(torch.float64)
        doc_accum /= N  # mean over MC samples → unbiased NELBO estimate

        for b in range(end - start):
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
