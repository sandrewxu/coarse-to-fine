"""Shared utilities for evaluation scripts.

The space tokenizer loader, vocab consistency check, bootstrap CI, and per-doc
JSONL writer are all reused across the AR and C2F evaluators.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.paths import PROJECT_ROOT


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
        tokenizer_dir = Path(c2f_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer"))
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
