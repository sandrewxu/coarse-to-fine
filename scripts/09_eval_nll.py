#!/usr/bin/env python3
"""
Unified NLL evaluator for C2F experiments (AR + C2F paths).

Reports nats/word on a held-out test set under a fixed tokenization. Every
model in the experiment sweep is scored through the repo's space tokenizer
(1 word = 1 token); the script fails hard on vocab-size mismatches so that
cross-model comparisons are not silently corrupted.

Model kinds
-----------
``ar``
    Standard autoregressive LM. Exact ``-log p(x)`` via shifted
    cross-entropy on raw text tokens.

``c2f``
    :class:`src.qwen3_joint.modeling.C2FForCausalLM`. Reports the C2F
    training objective (unshifted CE for block-mask, shifted for causal)
    as an **IWAE-1 lower bound** on the marginal ``log p(x)`` — i.e.
    the standard ELBO, using a gold ``z`` produced by ``q_φ`` and stored
    in the test parquet.

    Tighter IWAE-K (K > 1) requires sampling multiple ``z`` per document
    from ``q_φ`` and reweighting; that needs an inference server pass
    and is intentionally deferred. ``--K`` is validated to be 1 for
    ``c2f`` until that lands.

``diffusion``
    Not implemented. The R4 discrete-diffusion baseline will register
    a handler here when it is trained.

Outputs
-------
* Aggregate nats/word (equivalently nats/token, since space tokenizer is
  1:1) with a bootstrap 95 % CI over documents.
* Total scored tokens and document count.
* For ``c2f``: per-scale mean nats/token (``z_4, z_3, z_2, z_1, text``).
* Optional ``--out-jsonl``: one record per doc with
  ``{"idx", "nll", "num_tokens", "per_scale"}``.

Examples
--------
.. code-block:: bash

    python scripts/09_eval_nll.py \\
        --model-kind ar --ckpt checkpoints/ar/ \\
        --test data/tinystoriesv2_shuffled/tinystoriesv2.test.jsonl \\
        --limit 2000

    python scripts/09_eval_nll.py \\
        --model-kind c2f --ckpt checkpoints/decoder/ \\
        --test data/local_generations/c2f_test.parquet \\
        --K 1
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Shared helpers ────────────────────────────────────────────────────────


def _load_space_tokenizer(config: dict[str, Any], tokenizer_dir: Path | None = None):
    """Load the repo's space (word-level) tokenizer."""
    from src.c2f_training.tokenizer import load_or_train_space_tokenizer

    c2f_cfg = config.get("c2f_training", {})
    if tokenizer_dir is None:
        tokenizer_dir = Path(
            c2f_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
        )
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = PROJECT_ROOT / tokenizer_dir

    return load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=c2f_cfg.get("dataset_dir", "data/sft_dataset"),
        dataset_format=c2f_cfg.get("dataset_format", "sft"),
    )


def _check_vocab_consistency(model_vocab: int, tokenizer_vocab: int) -> None:
    """Abort on vocab mismatch — silent mismatch corrupts nats/word compares."""
    if model_vocab != tokenizer_vocab:
        raise RuntimeError(
            f"Vocab mismatch: model has {model_vocab}, tokenizer has "
            f"{tokenizer_vocab}. Cross-model nats/word comparisons require a "
            "shared tokenizer. Re-train the baseline with the space tokenizer "
            "at checkpoints/decoder/tokenizer."
        )


def _bootstrap_ci(
    per_doc_nll: np.ndarray,
    per_doc_tokens: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean nats/token, lo95, hi95) via token-weighted bootstrap over docs."""
    rng = np.random.default_rng(seed)
    n = len(per_doc_nll)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = per_doc_nll[idx].sum() / max(per_doc_tokens[idx].sum(), 1)
    point = per_doc_nll.sum() / max(per_doc_tokens.sum(), 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _save_per_doc(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ─── AR path ───────────────────────────────────────────────────────────────


def _load_ar_jsonl(path: Path, limit: int | None) -> list[str]:
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
def eval_ar(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Exact ``-log p(x)`` under a causal LM, via shifted CE on raw text."""
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenizer: repo's space tokenizer (1 word = 1 token).
    tokenizer = _load_space_tokenizer(config, args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"Loading AR model from {args.ckpt}...")
    model = AutoModelForCausalLM.from_pretrained(str(args.ckpt), trust_remote_code=True)
    model.to(device)
    model.eval()
    _check_vocab_consistency(model.config.vocab_size, tokenizer.vocab_size)

    docs = _load_ar_jsonl(args.test, args.limit)
    print(f"Scoring {len(docs)} documents...")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_doc_nll = np.empty(len(docs), dtype=np.float64)
    per_doc_tokens = np.empty(len(docs), dtype=np.int64)
    rows: list[dict] = []

    for i, text in enumerate(docs):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # BOS prefix so logits[0] can be used to predict the first word.
        bos = tokenizer.bos_token_id or tokenizer.eos_token_id
        ids = [bos] + ids
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]  # predict positions 1..T
        targets = input_ids[:, 1:]
        per_tok = loss_fn(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        nll = float(per_tok.sum().item())
        n_tok = int(targets.numel())
        per_doc_nll[i] = nll
        per_doc_tokens[i] = n_tok
        rows.append({"idx": i, "nll": nll, "num_tokens": n_tok})

    point, lo, hi = _bootstrap_ci(per_doc_nll, per_doc_tokens)
    return {
        "model_kind": "ar",
        "ckpt": str(args.ckpt),
        "num_docs": len(docs),
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "per_doc_rows": rows,
    }


# ─── C2F path ──────────────────────────────────────────────────────────────


def _build_c2f_dataset(
    test_path: Path, config: dict[str, Any], tokenizer
) -> Any:
    """Build a C2FDataset over the test parquet."""
    from src.c2f_training.dataset import C2FDataset
    import pyarrow.parquet as pq

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


def _scale_ranges(scale_lengths: list[int]) -> list[tuple[int, int]]:
    """(start, end) token slices per scale, relative to the BOS-prefixed sequence."""
    ranges: list[tuple[int, int]] = []
    pos = 1
    for length in scale_lengths:
        ranges.append((pos, pos + length))
        pos += length
    return ranges


@torch.no_grad()
def eval_c2f(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """ELBO (IWAE-1) on the joint ``p_θ(x, z)`` with gold ``z`` from the parquet."""
    from src.qwen3_joint.configuration import C2FConfig
    from src.qwen3_joint.modeling import C2FForCausalLM
    from src.rl.reward import _load_c2f_weights

    if args.K != 1:
        raise NotImplementedError(
            "IWAE-K > 1 for c2f requires sampling multiple z from q_φ per "
            "document; the sampler will be added in a follow-up. Re-run with "
            "--K 1 for the ELBO lower bound."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = _load_space_tokenizer(config, args.tokenizer_dir)

    print(f"Loading C2F model from {args.ckpt}...")
    model_config = C2FConfig.from_pretrained(str(args.ckpt))
    model = C2FForCausalLM(model_config)
    model = _load_c2f_weights(model, args.ckpt)
    model.to(device)
    model.eval()
    _check_vocab_consistency(model.config.vocab_size, tokenizer.vocab_size)

    dataset = _build_c2f_dataset(args.test, config, tokenizer)
    if args.limit is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    print(f"Scoring {len(dataset)} documents...")

    scale_names = ["z_4", "z_3", "z_2", "z_1", "text"]
    scale_ranges = _scale_ranges(config["scale_lengths"])
    mask_type = model.config.mask_type

    # Per-doc and per-scale accumulators.
    per_doc_nll = np.empty(len(dataset), dtype=np.float64)
    per_doc_tokens = np.empty(len(dataset), dtype=np.int64)
    per_scale_nll = {name: 0.0 for name in scale_names}
    per_scale_tokens = {name: 0 for name in scale_names}
    rows: list[dict] = []

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

        doc_nll = (per_tok * mask).sum(dim=1).cpu().numpy()  # [B]
        doc_tok = mask.sum(dim=1).cpu().numpy().astype(np.int64)

        for b in range(B):
            per_doc_nll[idx] = float(doc_nll[b])
            per_doc_tokens[idx] = int(doc_tok[b])
            per_doc_scales: dict[str, float] = {}
            for name, (start, end) in zip(scale_names, scale_ranges):
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

    point, lo, hi = _bootstrap_ci(per_doc_nll, per_doc_tokens)
    per_scale_mean = {
        name: (per_scale_nll[name] / per_scale_tokens[name])
        if per_scale_tokens[name] > 0
        else float("nan")
        for name in scale_names
    }
    return {
        "model_kind": "c2f",
        "ckpt": str(args.ckpt),
        "mask_type": mask_type,
        "K": args.K,
        "num_docs": len(dataset),
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "per_scale_nats_per_token": per_scale_mean,
        "per_doc_rows": rows,
    }


# ─── Diffusion path (stub) ─────────────────────────────────────────────────


def eval_diffusion(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(
        "Discrete-diffusion NLL will be added with the R4 baseline "
        "(src/baselines/diffusion.py). Report the ELBO bound at 128 "
        "denoising steps for comparability with C2F IWAE bounds."
    )


# ─── Main ──────────────────────────────────────────────────────────────────


def _print_summary(result: dict[str, Any]) -> None:
    mean = result["nats_per_word"]
    lo, hi = result["nats_per_word_ci95"]
    bits = mean / math.log(2)
    ppl = math.exp(mean)
    print()
    print("=" * 60)
    print(f"  model_kind       = {result['model_kind']}")
    print(f"  ckpt             = {result['ckpt']}")
    if "mask_type" in result:
        print(f"  mask_type        = {result['mask_type']}")
    if "K" in result:
        print(f"  IWAE K           = {result['K']}")
    print(f"  num_docs         = {result['num_docs']}")
    print(f"  total_tokens     = {result['total_tokens']}")
    print(f"  nats/word        = {mean:.4f}  (bits/word = {bits:.4f}, ppl = {ppl:.2f})")
    print(f"  95% CI (nats/w)  = [{lo:.4f}, {hi:.4f}]")
    if "per_scale_nats_per_token" in result:
        print("  per-scale nats/token:")
        for name, val in result["per_scale_nats_per_token"].items():
            print(f"    {name:>5s}: {val:.4f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified NLL evaluator for C2F experiments")
    parser.add_argument("--model-kind", required=True, choices=["ar", "c2f", "diffusion"])
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Test data. jsonl for --model-kind ar; parquet (text or prompt+response) for c2f.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of docs scored.")
    parser.add_argument("--batch-size", type=int, default=8, help="C2F forward batch size.")
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="IWAE sample count for c2f (only K=1 supported; K>1 needs q_φ sampler).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "latent_generation.yaml",
        help="Experiment YAML — used for scale_lengths, tokenizer_dir, etc.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Override the space tokenizer directory (else uses config).",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Optional per-doc JSONL output for downstream bootstrap / plotting.",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"Error: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 1
    if not args.test.exists():
        print(f"Error: test file not found: {args.test}", file=sys.stderr)
        return 1

    from src.config import load_config

    config = load_config(args.config)

    dispatch = {"ar": eval_ar, "c2f": eval_c2f, "diffusion": eval_diffusion}
    result = dispatch[args.model_kind](args, config)

    _print_summary(result)

    if args.out_jsonl is not None:
        _save_per_doc(args.out_jsonl, result["per_doc_rows"])
        print(f"Per-doc rows saved to {args.out_jsonl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
