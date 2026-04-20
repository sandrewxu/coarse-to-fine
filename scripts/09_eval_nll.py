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
from src.eval.bound import eval_c2f_bound
from src.eval.common import save_per_doc

log = get_logger(__name__)


def _print_summary(result: dict[str, Any]) -> None:
    mean = result["nats_per_word"]
    lo, hi = result["nats_per_word_ci95"]
    bits = mean / math.log(2)
    ppl = math.exp(mean) if mean < 700 else float("inf")
    log.info("=" * 60)
    log.info("  model_kind       = %s", result["model_kind"])
    log.info("  ckpt             = %s", result["ckpt"])
    if "sft_ckpt" in result:
        log.info("  sft_ckpt (q_φ)   = %s", result["sft_ckpt"])
    if "mask_type" in result:
        log.info("  mask_type        = %s", result["mask_type"])
    if "K" in result:
        log.info("  IWAE K           = %d", result["K"])
    if "N" in result:
        log.info("  MC samples N     = %d", result["N"])
    log.info("  num_docs         = %d", result["num_docs"])
    log.info("  total_tokens     = %d", result["total_tokens"])
    # For the c2f_bound path, "nats_per_word" is really
    # (joint_nll_p - nll_q) / text_word_count — a per-text-word upper bound
    # on -log p(x) that's directly comparable to AR's exact NLL.
    label = "nats/text_word (UB)" if result["model_kind"] == "c2f_bound" else "nats/word"
    log.info("  %-17s = %.4f  (bits/word = %.4f, ppl = %.4g)", label, mean, bits, ppl)
    log.info("  95%% CI           = [%.4f, %.4f]", lo, hi)
    if "mean_joint_nll_p" in result:
        log.info(
            "  mean joint NLL_p = %.4f  (mean NLL_q = %.4f)",
            result["mean_joint_nll_p"],
            result["mean_nll_q"],
        )
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
        help="Test data. Supported formats: .jsonl ({'text': ...} per line), "
        "SFT parquet (prompt + response columns), or c2f parquet (flattened "
        "text column). Passing the same c2f_val.parquet to all four "
        "--model-kind values gives an apples-to-apples comparison on the "
        "verified ~12k-doc subset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of docs scored.")
    parser.add_argument("--batch-size", type=int, default=8, help="C2F forward batch size.")
    parser.add_argument(
        "--ar-batch-size",
        type=int,
        default=16,
        help="AR doc batch size (right-padded + attention mask; 1 = legacy per-doc).",
    )
    parser.add_argument(
        "--mc-batch-size",
        type=int,
        default=16,
        help="Diffusion MC samples packed per forward; effective batch is "
        "--batch-size * --mc-batch-size (clamped to --N).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="IWAE sample count for c2f (only K=1 supported; K>1 needs q_φ sampler).",
    )
    parser.add_argument(
        "--sft-ckpt",
        type=Path,
        default=None,
        help="SFT checkpoint used as q_φ. When set with --model-kind c2f, switch "
        "to the ELBO upper-bound evaluator (IWAE-1 with gold z): reports a true "
        "-log p(x) / text_word_count upper bound, directly comparable to AR.",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=4,
        help="SFT forward-pass batch size for the bound evaluator (typically "
        "lower than --batch-size; SFT is Qwen3-4B on BPE-tokenized sequences).",
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
            batch_size=args.ar_batch_size,
        )
    elif args.model_kind == "c2f":
        if args.sft_ckpt is not None:
            if not args.sft_ckpt.exists():
                log.error("sft-ckpt not found: %s", args.sft_ckpt)
                return 1
            result = eval_c2f_bound(
                c2f_ckpt=args.ckpt,
                sft_ckpt=args.sft_ckpt,
                test=args.test,
                config=config,
                limit=args.limit,
                batch_size=args.batch_size,
                sft_batch_size=args.sft_batch_size,
                tokenizer_dir=args.tokenizer_dir,
            )
        else:
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
            mc_batch_size=args.mc_batch_size,
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
