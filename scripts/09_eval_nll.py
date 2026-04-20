#!/usr/bin/env python3
"""Step 9 — unified NLL evaluator for C2F experiments.

Reports nats/word (== nats/token under the project's space tokenizer) on a
held-out test set, with a bootstrap 95 % CI over documents. The script fails
hard on vocab-size mismatches so cross-model comparisons are not silently
corrupted.

Model kinds
-----------
``ar``
    Standard autoregressive LM. Exact ``-log p(x)`` via shifted CE on raw text.

``c2f``
    :class:`src.c2f_model.modeling.C2FForCausalLM`. Reports the C2F training
    objective (unshifted CE for block-mask, shifted for causal) as an IWAE-1
    lower bound on ``log p(x)``, using gold ``z`` from the test parquet.
    Tighter IWAE-K is intentionally deferred (``--K`` must be 1 for now).

``diffusion``
    Stub. The R4 discrete-diffusion baseline will register a handler here.

Examples
--------
.. code-block:: bash

    python scripts/09_eval_nll.py \\
        --model-kind ar --ckpt checkpoints/ar/ \\
        --test data/tinystoriesv2_shuffled/tinystoriesv2.test.jsonl \\
        --limit 2000

    python scripts/09_eval_nll.py \\
        --model-kind c2f --ckpt checkpoints/decoder/ \\
        --test data/local_generations/c2f_test.parquet --K 1
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT
from src.eval import eval_ar, eval_c2f, eval_diffusion
from src.eval.common import save_per_doc

log = get_logger(__name__)


def _print_summary(result: dict[str, Any]) -> None:
    mean = result["nats_per_word"]
    lo, hi = result["nats_per_word_ci95"]
    bits = mean / math.log(2)
    ppl = math.exp(mean)
    log.info("=" * 60)
    log.info("  model_kind       = %s", result["model_kind"])
    log.info("  ckpt             = %s", result["ckpt"])
    if "mask_type" in result:
        log.info("  mask_type        = %s", result["mask_type"])
    if "K" in result:
        log.info("  IWAE K           = %d", result["K"])
    if "N" in result:
        log.info("  MC samples N     = %d", result["N"])
    log.info("  num_docs         = %d", result["num_docs"])
    log.info("  total_tokens     = %d", result["total_tokens"])
    log.info("  nats/word        = %.4f  (bits/word = %.4f, ppl = %.2f)", mean, bits, ppl)
    log.info("  95%% CI (nats/w)  = [%.4f, %.4f]", lo, hi)
    if "per_scale_nats_per_token" in result:
        log.info("  per-scale nats/token:")
        for name, val in result["per_scale_nats_per_token"].items():
            log.info("    %5s: %.4f", name, val)
    log.info("=" * 60)


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
        "--N",
        type=int,
        default=128,
        help="MC sample count per doc for diffusion NELBO (analog of --K).",
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
        log.error("ckpt not found: %s", args.ckpt)
        return 1
    if not args.test.exists():
        log.error("test file not found: %s", args.test)
        return 1

    from src.config import load_config

    config = load_config(args.config)

    if args.model_kind == "ar":
        result = eval_ar(
            ckpt=args.ckpt,
            test=args.test,
            config=config,
            limit=args.limit,
            tokenizer_dir=args.tokenizer_dir,
        )
    elif args.model_kind == "c2f":
        result = eval_c2f(
            ckpt=args.ckpt,
            test=args.test,
            config=config,
            limit=args.limit,
            batch_size=args.batch_size,
            tokenizer_dir=args.tokenizer_dir,
            K=args.K,
        )
    elif args.model_kind == "diffusion":
        result = eval_diffusion(
            ckpt=args.ckpt,
            test=args.test,
            config=config,
            limit=args.limit,
            tokenizer_dir=args.tokenizer_dir,
            N=args.N,
            batch_size=args.batch_size,
        )
    else:  # pragma: no cover — argparse choices guard this
        raise ValueError(args.model_kind)

    _print_summary(result)

    if args.out_jsonl is not None:
        save_per_doc(args.out_jsonl, result["per_doc_rows"])
        log.info("Per-doc rows saved to %s", args.out_jsonl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
