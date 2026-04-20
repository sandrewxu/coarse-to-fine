"""Autoregressive-LM NLL evaluator.

Exact ``-log p(x)`` via shifted cross-entropy on raw text tokens — no latent
variables. Tokenizer is the project's space tokenizer (1 word = 1 token), so
nats/word == nats/token and is directly comparable to the C2F evaluator.

Documents are right-padded to the per-batch max length and scored in parallel:
``attention_mask`` stops real positions from attending to pad keys, and pad
target positions are set to ``-100`` so they drop out of the CE sum. This gives
per-doc NLLs that match the unbatched (``batch_size=1``) reference to within
floating-point noise — see ``tests/test_eval_batching.py``.
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
    """Load up to ``limit`` documents from a JSONL file (one ``{"text": str}`` per line)."""
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
def eval_ar(
    *,
    ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None = None,
    tokenizer_dir: Path | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Compute exact NLL of a causal LM on a JSONL test set.

    Args:
        batch_size: Documents packed per forward pass. Right-padded to the
            per-batch max length; pad targets are ignored in the CE. Set to 1
            to recover the original one-doc-at-a-time behavior.

    Returns:
        Dict with ``model_kind``, ``ckpt``, ``num_docs``, ``total_tokens``,
        ``nats_per_word``, ``nats_per_word_ci95``, and a ``per_doc_rows`` list
        with one ``{"idx", "nll", "num_tokens"}`` record per scored document.
    """
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = load_space_tokenizer(config, tokenizer_dir)

    log.info("Loading AR model from %s...", ckpt)
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True)
    model.to(device)
    model.eval()
    check_vocab_consistency(model.config.vocab_size, tokenizer.vocab_size)

    docs = _load_jsonl(test, limit)
    log.info("Scoring %d documents (batch_size=%d)...", len(docs), batch_size)

    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id for padding.")

    # Pre-tokenize once so the batching loop only does padding + forward.
    tokenized: list[list[int]] = []
    for text in docs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokenized.append([bos, *ids])

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_doc_nll = np.empty(len(docs), dtype=np.float64)
    per_doc_tokens = np.empty(len(docs), dtype=np.int64)
    rows: list[dict] = []

    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        batch_ids = tokenized[start:end]
        max_len = max(len(ids) for ids in batch_ids)
        B = end - start

        input_ids = torch.full((B, max_len), pad, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for b, ids in enumerate(batch_ids):
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[b, : len(ids)] = 1

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]  # predict positions 1..T
        targets = input_ids[:, 1:].clone()
        # A target at shifted position t (= original position t+1) is only scored
        # if attn_mask[:, t+1] == 1; everything else becomes ignore_index.
        target_mask = attn_mask[:, 1:].bool()
        targets = targets.masked_fill(~target_mask, -100)

        _, T, V = logits.shape
        per_tok = loss_fn(logits.reshape(-1, V), targets.reshape(-1)).view(B, T)
        # ignore_index zeros out pad contributions, so a plain sum is correct.
        doc_nll = per_tok.sum(dim=1).cpu().numpy()
        doc_tok = target_mask.sum(dim=1).cpu().numpy().astype(np.int64)

        for b in range(B):
            idx = start + b
            nll = float(doc_nll[b])
            n_tok = int(doc_tok[b])
            per_doc_nll[idx] = nll
            per_doc_tokens[idx] = n_tok
            rows.append({"idx": idx, "nll": nll, "num_tokens": n_tok})

    point, lo, hi = bootstrap_ci(per_doc_nll, per_doc_tokens)
    return {
        "model_kind": "ar",
        "ckpt": str(ckpt),
        "num_docs": len(docs),
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "per_doc_rows": rows,
    }
