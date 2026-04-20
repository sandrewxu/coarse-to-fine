"""Shared utilities for evaluation scripts.

The space tokenizer loader, vocab consistency check, bootstrap CI, and per-doc
JSONL writer are all reused across the AR and C2F evaluators.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.paths import PROJECT_ROOT


def load_test_docs(
    test: Path,
    limit: int | None,
    *,
    text_word_count: int,
) -> list[str]:
    """Load raw text documents from any of the test-file formats we support.

    Returns one text string per document (the ``x`` itself — no latents, no
    chat wrapping). Used by the AR and diffusion evaluators so they score the
    same doc subset the C2F evaluators do when handed the same parquet.

    Dispatch rules:
    - ``.jsonl``: one JSON object per line; read the ``"text"`` field (or the
      raw line if not JSON).
    - ``.parquet`` with a ``prompt`` column: SFT or veRL format; return
      ``prompt`` (the raw text).
    - ``.parquet`` with only a ``text`` column: c2f (flattened) format; the
      final ``text_word_count`` whitespace-separated words of each row are
      the original ``x`` by construction of ``flatten_for_c2f``.

    ``limit`` (if not None) is applied after loading.
    """
    suffix = test.suffix.lower()
    if suffix == ".jsonl":
        docs: list[str] = []
        with test.open() as f:
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

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        cols = pq.read_schema(str(test)).names
        if "prompt" in cols:
            prompts = pq.read_table(str(test), columns=["prompt"]).column("prompt").to_pylist()
            # veRL format stores prompt as a list of chat messages; unwrap to text.
            flat: list[str] = []
            for p in prompts:
                if isinstance(p, list) and p and isinstance(p[0], dict):
                    flat.append(p[-1].get("content", ""))
                else:
                    flat.append(p if isinstance(p, str) else str(p))
            return flat[:limit] if limit is not None else flat
        if "text" in cols:
            texts = pq.read_table(str(test), columns=["text"]).column("text").to_pylist()
            out = [" ".join(t.split()[-text_word_count:]) for t in texts]
            return out[:limit] if limit is not None else out
        raise ValueError(
            f"Parquet {test} has neither a 'prompt' nor 'text' column "
            f"(columns: {cols}); cannot load documents for evaluation."
        )

    raise ValueError(f"Unsupported test-file suffix {suffix!r} at {test}. Use .jsonl or .parquet.")


def load_space_tokenizer(config: dict[str, Any], tokenizer_dir: Path | None = None):
    """Load the repo's space (word-level) tokenizer.

    Args:
        config: Experiment config (uses ``c2f_training.tokenizer_dir`` /
            ``dataset_dir`` / ``dataset_format`` for fallback).
        tokenizer_dir: Explicit override for the tokenizer directory.

    Returns:
        A ``PreTrainedTokenizerFast`` ready for ``.encode``.
    """
    from src.c2f_model.training.tokenizer import load_or_train_space_tokenizer

    c2f_cfg = config.get("c2f_training", {})
    if tokenizer_dir is None:
        tokenizer_dir = Path(c2f_cfg.get("tokenizer_dir", "checkpoints/tokenizer"))
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = PROJECT_ROOT / tokenizer_dir

    return load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=c2f_cfg.get("dataset_dir", "data/sft_dataset"),
        dataset_format=c2f_cfg.get("dataset_format", "sft"),
    )


def check_vocab_consistency(model_vocab: int, tokenizer_vocab: int) -> None:
    """Abort on vocab mismatch — silent mismatch corrupts nats/word compares."""
    if model_vocab != tokenizer_vocab:
        raise RuntimeError(
            f"Vocab mismatch: model has {model_vocab}, tokenizer has "
            f"{tokenizer_vocab}. Cross-model nats/word comparisons require a "
            "shared tokenizer. Re-train the baseline with the space tokenizer "
            "at checkpoints/decoder/tokenizer."
        )


def bootstrap_ci(
    per_doc_nll: np.ndarray,
    per_doc_tokens: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean nats/token, lo95, hi95)`` via token-weighted bootstrap over docs.

    The ``n_resamples`` and ``seed`` defaults are fixed across all experiment runs
    for inter-run comparability — don't change them without invalidating
    historical comparisons.
    """
    rng = np.random.default_rng(seed)
    n = len(per_doc_nll)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = per_doc_nll[idx].sum() / max(per_doc_tokens[idx].sum(), 1)
    point = per_doc_nll.sum() / max(per_doc_tokens.sum(), 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def save_per_doc(out_path: Path, rows: list[dict]) -> None:
    """Write one JSON-encoded row per line to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
