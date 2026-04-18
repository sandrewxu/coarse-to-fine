#!/usr/bin/env python3
"""
Submit, monitor, and download OpenAI Batch API jobs.

Usage:
    # Submit a batch request
    python scripts/02_submit_batch.py \
        --input data/prompt_data/.../sft.jsonl \
        --config config/latent_generation.yaml

    # Submit and poll until complete
    python scripts/02_submit_batch.py \
        --input data/prompt_data/.../sft.jsonl \
        --config config/latent_generation.yaml \
        --monitor

    # Download all completed batches (no submission)
    python scripts/02_submit_batch.py \
        --download \
        --run-tag latent_generation_10k_v1

    # Monitor all active batches
    python scripts/02_submit_batch.py --monitor-all
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.batch.client import create_client
from src.batch.submit import download_completed, monitor_all, monitor_batch, submit_batch
from src.common.env import load_env
from src.common.logging import get_logger
from src.config import load_config

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Submit, monitor, and download OpenAI batch jobs")
    parser.add_argument("--input", type=Path, default=None, help="Request JSONL file to submit")
    parser.add_argument("--config", type=Path, default=None, help="Experiment config YAML")
    parser.add_argument("--model", type=str, default=None, help="Model name for metadata")
    parser.add_argument(
        "--run-tag", type=str, default=None, help="Run tag for batch metadata/filtering"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory for downloaded outputs"
    )
    parser.add_argument(
        "--extra-metadata", type=str, default=None, help="Additional metadata as JSON string"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Poll until submitted batch completes"
    )
    parser.add_argument(
        "--monitor-all", action="store_true", help="Monitor all active batches (no submission)"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download completed batches (no submission)"
    )
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    load_env()

    # Load config (if provided)
    batch_config: dict = {}
    if args.config:
        config = load_config(args.config)
        batch_config = config.get("batch", {})

    model = args.model or batch_config.get("model", "gpt-5-nano-2025-08-07")
    run_tag = args.run_tag or batch_config.get("run_tag", "latent_generation_10k_v1")
    output_dir = args.output_dir or Path(batch_config.get("output_dir", "data/batch_outputs"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    client = create_client()

    # Download-only mode
    if args.download:
        download_completed(client, output_dir, run_tag=run_tag)
        return 0

    # Monitor-all mode
    if args.monitor_all:
        monitor_all(client, output_dir, poll_interval=args.poll_interval)
        return 0

    # Submit mode
    if not args.input:
        log.error("--input is required for submission (or use --download / --monitor-all)")
        sys.exit(1)

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    # Build metadata
    metadata = {"model": model, "run_tag": run_tag}

    # Add batch config fields as metadata
    for key in ("reasoning_effort", "verbosity", "system_prompt", "user_prompt"):
        value = batch_config.get(key)
        if value:
            metadata[key] = value

    if args.extra_metadata:
        try:
            metadata.update(json.loads(args.extra_metadata))
        except json.JSONDecodeError:
            log.error("--extra-metadata must be valid JSON")
            sys.exit(1)

    batch_id = submit_batch(client, args.input, metadata=metadata)

    if args.monitor:
        log.error(f"\nMonitoring batch {batch_id}...")
        result = monitor_batch(client, batch_id, output_dir, poll_interval=args.poll_interval)
        if result:
            log.info(f"\nOutput saved to: {result}")
        else:
            log.info("\nBatch did not complete successfully.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
