#!/usr/bin/env python3
"""
Verify batch API outputs and write veRL-compatible SFT parquet.

Usage:
    python scripts/03_verify_outputs.py \
        --input data/batch_outputs/.../output.jsonl \
        --config config/latent_generation.yaml \
        --prompts data/prompt_data/.../sft.jsonl \
        --output data/verified/latent_generation_10k_v1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import get_logger
from src.config import load_config
from src.data.schemas import VerificationStats
from src.sft.dataset import create_and_save_dataset
from src.verification import VerificationResult, verify

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider-specific extraction
# ---------------------------------------------------------------------------


def extract_openai_batch_line(line: str) -> tuple[str, str, str | None]:
    """
    Extract (custom_id, content, error) from an OpenAI Batch API JSONL line.

    Returns:
        Tuple of (custom_id, content, error_message_or_None)
    """
    data = json.loads(line)
    custom_id = data.get("custom_id", "")
    error = data.get("error")
    response = data.get("response", {})
    status_code = response.get("status_code")

    if error or (status_code and status_code != 200):
        err_msg = str(error) if error else f"status_code={status_code}"
        return custom_id, "", err_msg

    choices = response.get("body", {}).get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "")

    return custom_id, content, None


def extract_prompts_from_sft_jsonl(path: Path) -> dict[str, str]:
    """
    Extract original text prompts from sft.jsonl (OpenAI Batch API request format).
    Returns dict mapping custom_id -> last user message content.
    """
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            messages = data.get("body", {}).get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    prompts[custom_id] = msg.get("content", "")
                    break
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify batch outputs and create training dataset")
    parser.add_argument("--input", required=True, type=Path, help="Batch output JSONL file")
    parser.add_argument("--config", required=True, type=Path, help="Experiment config YAML")
    parser.add_argument(
        "--prompts", type=Path, default=None, help="sft.jsonl with original prompts"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output dir for verification stats"
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    word_count_constraints = config["word_count_constraints"]
    strict = config.get("verification", {}).get("strict_word_count", True)
    provider = config.get("batch", {}).get("provider", "openai")

    if provider != "openai":
        log.error("Unsupported provider %r. Only 'openai' is supported.", provider)
        sys.exit(1)

    output_dir = Path(
        args.output or config.get("dataset", {}).get("output_dir", "data/verified/output")
    )

    # Load prompts
    prompts_path = args.prompts
    if prompts_path is None:
        p = config.get("sft", {}).get("prompt_data")
        if p:
            prompts_path = Path(p)
            if not prompts_path.is_absolute():
                prompts_path = Path(__file__).resolve().parent.parent / prompts_path

    prompts_by_id: dict[str, str] = {}
    if prompts_path and prompts_path.exists():
        log.info(f"Loading prompts from {prompts_path}...")
        prompts_by_id = extract_prompts_from_sft_jsonl(prompts_path)
        log.info(f"  Found {len(prompts_by_id)} prompts")
    else:
        log.warning("No prompts file found; prompt column will be empty.")

    # Process and verify
    log.error(f"Processing {args.input}...")
    stats = VerificationStats()
    results: list[VerificationResult] = []

    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                custom_id, content, error = extract_openai_batch_line(line)
            except Exception as e:
                log.warning(f"Failed to parse line {line_num}: {e}")
                continue

            stats.total_processed += 1

            if error:
                result = VerificationResult(custom_id=custom_id, passed=False, raw_content="")
                result.fail(f"API error: {error}")
            else:
                result = verify(
                    content, word_count_constraints, custom_id=custom_id, strict_word_count=strict
                )

            results.append(result)
            if result.passed:
                stats.record_pass()
            else:
                for reason in result.failure_reasons:
                    stats.record_failure(reason)

            if line_num % 1000 == 0:
                log.error(f"  {line_num} lines... ({stats.passed} passed, {stats.failed} failed)")

    # Report
    log.info(f"\n{'=' * 60}")
    log.info(stats)
    log.info(f"{'=' * 60}\n")

    if stats.passed == 0:
        log.error("No outputs passed verification.")
        sys.exit(1)

    # Save SFT dataset
    sft_dir = config.get("sft", {}).get("dataset_dir", "data/sft_dataset")
    sft_output = (
        Path(sft_dir)
        if Path(sft_dir).is_absolute()
        else Path(__file__).resolve().parent.parent / sft_dir
    )
    log.info(f"Creating SFT parquet from {stats.passed} verified outputs...")
    dataset, saved_path = create_and_save_dataset(
        verified_results=results,
        output_dir=sft_output,
        prompts_by_id=prompts_by_id,
    )
    log.info(f"  Saved to: {saved_path} ({len(dataset)} examples)")

    # Save stats
    stats_path = output_dir / "verification_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_processed": stats.total_processed,
                "passed": stats.passed,
                "failed": stats.failed,
                "pass_rate": stats.pass_rate,
                "failure_breakdown": stats.failure_breakdown,
            },
            f,
            indent=2,
        )
    log.info(f"  Stats saved to: {stats_path}")
    log.info("\nDone!")


if __name__ == "__main__":
    main()
