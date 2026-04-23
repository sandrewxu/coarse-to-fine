#!/usr/bin/env python3
"""Step 9 — unified NLL evaluator for C2F experiments.

Reports a bootstrap 95 % CI over documents. The script fails hard on
vocab-size mismatches so cross-model comparisons are not silently corrupted.

Which model kinds are directly comparable
-----------------------------------------
Three evaluators report ``-log p(x) / text_word_count`` (nats per text word,
either exactly or as an upper bound) and are **directly comparable to each
other**:

``ar``
    Standard autoregressive LM. Exact ``-log p(x)`` via shifted CE on raw text.

``diffusion``
    R4 discrete-diffusion (MDLM/SUBS) baseline. MC upper bound with ``N`` samples.

``c2f-bound``
    C2F with the ``q_φ`` (SFT) correction term: IWAE-1 ELBO upper bound on
    ``-log p(x) / text_word_count``. Requires ``--sft-ckpt``.

One evaluator is **not comparable to the above** and is intended for C2F-vs-C2F
(architecture) diagnostics only:

``c2f-train-loss``
    C2F joint training objective: ``-log p(x, z_gold) / num_content_tokens``.
    Different numerator (joint vs marginal) *and* different denominator (62
    content tokens vs 32 text words). Use this to compare AR-C2F vs Block-C2F
    or to watch for training-loss regressions — **never** against the AR or
    diffusion baselines.

The deprecated ``c2f`` alias errors with a hint pointing to the two split
modes.

Examples
--------
.. code-block:: bash

    python scripts/09_eval_nll.py \\
        --model-kind ar --ckpt checkpoints/ar/ \\
        --test data/tinystoriesv2_shuffled/tinystoriesv2.test.jsonl \\
        --limit 2000

    python scripts/09_eval_nll.py \\
        --model-kind c2f-bound --ckpt checkpoints/decoder/ \\
        --sft-ckpt checkpoints/sft/ \\
        --test data/local_generations/c2f_test.parquet

    python scripts/09_eval_nll.py \\
        --model-kind c2f-train-loss --ckpt checkpoints/decoder/ \\
        --test data/local_generations/c2f_test.parquet
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
    comparable = result.get("comparable_to", "unknown")
    is_train_loss = comparable == "joint_train_loss_per_token"

    if is_train_loss:
        mean = result["nats_per_joint_token"]
        lo, hi = result["nats_per_joint_token_ci95"]
        label = "nats/joint_token"
    else:
        mean = result["nats_per_word"]
        lo, hi = result["nats_per_word_ci95"]
        label = "nats/text_word (UB)" if result["model_kind"] == "c2f_bound" else "nats/text_word"

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
    log.info("  %-17s = %.4f  (bits = %.4f, exp = %.4g)", label, mean, bits, ppl)
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
    if is_train_loss:
        log.warning(
            "NOTE: this is the C2F joint training loss (nats per joint content "
            "token). It is NOT comparable to --model-kind ar / diffusion / "
            "c2f-bound — different numerator (joint vs marginal) and different "
            "denominator (content tokens vs text words). For a number you can "
            "put on the same axis as AR/diffusion, re-run with "
            "--model-kind c2f-bound --sft-ckpt <path>."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified NLL evaluator for C2F experiments")
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=["ar", "diffusion", "c2f-bound", "c2f-train-loss", "c2f"],
        help="'ar' / 'diffusion' / 'c2f-bound' all report -log p(x) per text "
        "word and are directly comparable. 'c2f-train-loss' reports the joint "
        "training objective and is for C2F-vs-C2F diagnostics only. 'c2f' is a "
        "deprecated alias and will error out with a hint.",
    )
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
        help="SFT checkpoint used as q_φ. Required for --model-kind c2f-bound. "
        "Unused by other modes.",
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

    if args.model_kind == "c2f":
        log.error(
            "--model-kind c2f is deprecated. Use 'c2f-bound' (ELBO upper bound "
            "on -log p(x)/text_word_count, requires --sft-ckpt; directly "
            "comparable to ar/diffusion) or 'c2f-train-loss' (joint training "
            "objective per content token, for C2F-vs-C2F diagnostics only)."
        )
        return 2

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
    elif args.model_kind == "c2f-bound":
        if args.sft_ckpt is None:
            log.error(
                "--model-kind c2f-bound requires --sft-ckpt (the q_φ model). "
                "Without q_φ there is no correction term and the number is not "
                "an ELBO. For the uncorrected training-loss report, use "
                "--model-kind c2f-train-loss."
            )
            return 2
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
    elif args.model_kind == "c2f-train-loss":
        if args.sft_ckpt is not None:
            log.warning(
                "--sft-ckpt is ignored for --model-kind c2f-train-loss. "
                "Did you mean --model-kind c2f-bound?"
            )
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
