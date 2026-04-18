"""Autoregressive-LM NLL evaluator.

Exact ``-log p(x)`` via shifted cross-entropy on raw text tokens — no latent
variables. Tokenizer is the project's space tokenizer (1 word = 1 token), so
nats/word == nats/token and is directly comparable to the C2F evaluator.
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
) -> dict[str, Any]:
    """Compute exact NLL of a causal LM on a JSONL test set.

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
    log.info("Scoring %d documents...", len(docs))

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_doc_nll = np.empty(len(docs), dtype=np.float64)
    per_doc_tokens = np.empty(len(docs), dtype=np.int64)
    rows: list[dict] = []

    for i, text in enumerate(docs):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # BOS prefix so logits[0] can be used to predict the first word.
        bos = tokenizer.bos_token_id or tokenizer.eos_token_id
        ids = [bos, *ids]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]  # predict positions 1..T
        targets = input_ids[:, 1:]
        per_tok = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        nll = float(per_tok.sum().item())
        n_tok = int(targets.numel())
        per_doc_nll[i] = nll
        per_doc_tokens[i] = n_tok
        rows.append({"idx": i, "nll": nll, "num_tokens": n_tok})

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
