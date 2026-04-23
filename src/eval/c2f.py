"""C2F joint-model *training-loss* evaluator.

Reports ``-log p(x, z_gold) / num_content_tokens`` — the same quantity the C2F
model minimizes during training, computed on a held-out test set. Useful for
comparing **C2F variants to each other** (e.g. AR-C2F vs Block-C2F) and for
catching training-loss regressions; **not** directly comparable to the AR or
diffusion baselines — those report ``-log p(x) / text_word_count``, which has
both a different numerator (marginal vs joint) and a different denominator
(text words vs all content tokens including z).

For a true ELBO upper bound on ``-log p(x) / text_word_count`` that *is*
comparable to AR/diffusion, use :func:`src.eval.bound.eval_c2f_bound`, which
adds the ``-log q_φ(z|x)`` correction term required by the IWAE identity.

This evaluator's output dict carries ``comparable_to="joint_train_loss_per_token"``
so downstream plotting code can refuse to plot it against AR/diffusion.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.logging import get_logger
from src.eval.common import bootstrap_ci, check_vocab_consistency, load_space_tokenizer

log = get_logger(__name__)


def _scale_ranges(scale_lengths: list[int]) -> list[tuple[int, int]]:
    """``(start, end)`` token slices per scale, relative to the BOS-prefixed sequence."""
    ranges: list[tuple[int, int]] = []
    pos = 1
    for length in scale_lengths:
        ranges.append((pos, pos + length))
        pos += length
    return ranges


def _build_dataset(test_path: Path, config: dict[str, Any], tokenizer):
    """Build a ``C2FDataset`` over the test parquet (auto-detects c2f vs sft)."""
    import pyarrow.parquet as pq

    from src.c2f_model.training.dataset import C2FDataset

    cols = pq.read_schema(str(test_path)).names
    if "text" in cols:
        fmt = "c2f"
    elif "prompt" in cols and "response" in cols:
        fmt = "sft"
    else:
        raise ValueError(
            f"Cannot detect format from {test_path} (columns: {cols}). "
            "Expected 'text' (c2f) or 'prompt'+'response' (sft)."
        )

    return C2FDataset(
        data_dir=str(test_path.parent),
        tokenizer=tokenizer,
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config["word_count_constraints"],
        text_word_count=config.get("text_word_count", 32),
        parquet_filename=test_path.name,
        dataset_format=fmt,
    )


@torch.no_grad()
def eval_c2f(
    *,
    ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None = None,
    batch_size: int = 8,
    tokenizer_dir: Path | None = None,
    K: int = 1,
) -> dict[str, Any]:
    """Compute the IWAE-1 ELBO bound on ``log p(x)`` for the C2F model.

    Args:
        K: IWAE sample count. Only ``K=1`` is currently supported; ``K>1`` will
            require sampling ``z`` from ``q_φ`` per document (deferred).
    """
    from src.c2f_model.configuration import C2FConfig
    from src.c2f_model.modeling import C2FForCausalLM
    from src.rl.common import load_c2f_weights

    if K != 1:
        raise NotImplementedError(
            "IWAE-K > 1 for c2f requires sampling multiple z from q_φ per "
            "document; the sampler will be added in a follow-up. Re-run with "
            "K=1 for the ELBO lower bound."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = load_space_tokenizer(config, tokenizer_dir)

    log.info("Loading C2F model from %s...", ckpt)
    model_config = C2FConfig.from_pretrained(str(ckpt))
    model = C2FForCausalLM(model_config)
    model = load_c2f_weights(model, ckpt)
    model.to(device)
    model.eval()
    check_vocab_consistency(model.config.vocab_size, tokenizer.vocab_size)

    dataset = _build_dataset(test, config, tokenizer)
    if limit is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(limit, len(dataset))))

    log.info("Scoring %d documents...", len(dataset))

    scale_names = ["z_4", "z_3", "z_2", "z_1", "text"]
    scale_ranges = _scale_ranges(config["scale_lengths"])
    mask_type = model.config.mask_type

    per_doc_nll = np.empty(len(dataset), dtype=np.float64)
    per_doc_tokens = np.empty(len(dataset), dtype=np.int64)
    per_scale_nll = {name: 0.0 for name in scale_names}
    per_scale_tokens = {name: 0 for name in scale_names}
    rows: list[dict] = []

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids)
        logits = out.logits  # [B, T, V]

        if mask_type == "causal":
            logits = logits[:, :-1, :]
            lab = labels[:, 1:]
        else:
            lab = labels

        B, T, V = logits.shape
        per_tok = loss_fn(logits.reshape(-1, V), lab.reshape(-1)).view(B, T)
        mask = (lab != -100).float()

        doc_nll = (per_tok * mask).sum(dim=1).cpu().numpy()
        doc_tok = mask.sum(dim=1).cpu().numpy().astype(np.int64)

        for b in range(B):
            per_doc_nll[idx] = float(doc_nll[b])
            per_doc_tokens[idx] = int(doc_tok[b])
            per_doc_scales: dict[str, float] = {}
            for name, (start, end) in zip(scale_names, scale_ranges, strict=False):
                if mask_type == "causal":
                    s_slice = slice(start - 1, end - 1)
                else:
                    s_slice = slice(start, end)
                seg_tok = per_tok[b, s_slice] * mask[b, s_slice]
                seg_n = int(mask[b, s_slice].sum().item())
                seg_nll = float(seg_tok.sum().item())
                per_scale_nll[name] += seg_nll
                per_scale_tokens[name] += seg_n
                per_doc_scales[name] = seg_nll
            rows.append(
                {
                    "idx": idx,
                    "nll": float(doc_nll[b]),
                    "num_tokens": int(doc_tok[b]),
                    "per_scale_nll": per_doc_scales,
                }
            )
            idx += 1

    point, lo, hi = bootstrap_ci(per_doc_nll, per_doc_tokens)
    per_scale_mean = {
        name: per_scale_nll[name] / per_scale_tokens[name]
        if per_scale_tokens[name] > 0
        else float("nan")
        for name in scale_names
    }
    return {
        "model_kind": "c2f",
        "ckpt": str(ckpt),
        "mask_type": mask_type,
        "K": K,
        "num_docs": len(dataset),
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_joint_token": point,
        "nats_per_joint_token_ci95": [lo, hi],
        "denominator": "joint_content_tokens",
        "comparable_to": "joint_train_loss_per_token",
        "per_scale_nats_per_token": per_scale_mean,
        "per_doc_rows": rows,
    }
