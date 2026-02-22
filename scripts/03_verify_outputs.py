#!/usr/bin/env python3
"""
Verify batch API outputs and write veRL-compatible SFT parquet.

Writes SFT data to config's sft.dataset_dir (default data/sft_dataset/train.parquet)
with columns prompt (original text) and response (raw z_n: format latent layers).
Saves verification stats to --output.

Usage:
    python scripts/03_verify_outputs.py \
        --input data/batch_outputs/.../output.jsonl \
        --config config/experiments/latent_generation.yaml \
        --prompts data/prompt_data/.../sft.jsonl \
        --output data/verified/latent_generation_10k_v1
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schemas import BatchOutputItem, VerificationStats
from src.sft.dataset import create_and_save_dataset
from src.verification.rule_based import RuleBasedVerifier


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_prompts_from_sft_jsonl(sft_jsonl_path: Path) -> dict[str, str]:
    """
    Extract original text prompts from sft.jsonl (OpenAI Batch API request format).

    Each line has custom_id and body.messages. The last message with role 'user'
    is the original 32-word document.

    Args:
        sft_jsonl_path: Path to sft.jsonl file

    Returns:
        Dict mapping custom_id -> original text prompt
    """
    prompts = {}
    with open(sft_jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            custom_id = data.get('custom_id', '')
            messages = data.get('body', {}).get('messages', [])

            # Find the last user message (the actual document to summarize)
            prompt = ""
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    prompt = msg.get('content', '')
                    break

            prompts[custom_id] = prompt

    return prompts


def parse_batch_output_line(line: str) -> BatchOutputItem | None:
    """
    Parse a single JSONL line into a BatchOutputItem.

    Args:
        line: JSON string from output.jsonl

    Returns:
        BatchOutputItem or None if parsing failed
    """
    try:
        data = json.loads(line)

        # Extract relevant fields from nested structure
        custom_id = data.get('custom_id', '')
        response = data.get('response', {})
        error = data.get('error')

        # Extract content from response body
        status_code = response.get('status_code')
        body = response.get('body', {})
        choices = body.get('choices', [])

        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            content = message.get('content', '')
        else:
            content = ''

        model = body.get('model')

        return BatchOutputItem(
            custom_id=custom_id,
            content=content,
            status_code=status_code,
            error=error,
            model=model,
        )
    except Exception as e:
        print(f"Error parsing line: {e}", file=sys.stderr)
        return None


def process_batch_file(
    input_path: Path,
    verifier: RuleBasedVerifier,
    stats: VerificationStats,
) -> list:
    """
    Process batch output file line by line.

    Args:
        input_path: Path to output.jsonl file
        verifier: Configured verifier instance
        stats: Statistics tracker

    Returns:
        List of verification results
    """
    results = []

    print(f"Processing {input_path}...")
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse batch output
            item = parse_batch_output_line(line)
            if item is None:
                print(f"Warning: Failed to parse line {line_num}", file=sys.stderr)
                continue

            # Verify
            result = verifier.verify(item)
            results.append(result)

            # Update statistics
            stats.total_processed += 1
            if result.passed:
                stats.record_pass()
            else:
                for reason in result.failure_reasons:
                    stats.record_failure(reason)

            # Progress indicator
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines... ({stats.passed} passed, {stats.failed} failed)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify batch outputs and create training dataset"
    )
    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help="Path to batch output JSONL file"
    )
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help="Path to experiment configuration YAML"
    )
    parser.add_argument(
        '--prompts',
        type=Path,
        default=None,
        help="Path to sft.jsonl with original prompts (extracts last user message per custom_id)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help="Output directory for verification stats (overrides config)"
    )
    parser.add_argument(
        '--batch-id',
        type=str,
        default=None,
        help="Batch identifier for tracking"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Determine output directory
    output_dir = args.output or config.get('dataset', {}).get('output_dir', 'data/verified/output')
    output_path = Path(output_dir)

    # Extract batch_id from path if not provided
    batch_id = args.batch_id
    if batch_id is None:
        # Try to extract from path (e.g., batch_6971db0456e081908c39483dd5230333)
        for part in args.input.parts:
            if part.startswith('batch_'):
                batch_id = part
                break

    # Load prompts from sft.jsonl if provided, else try config path
    prompts_by_id = None
    prompts_path = args.prompts
    if prompts_path is None:
        # Check config for sft.prompt_data
        prompts_path_str = config.get('sft', {}).get('prompt_data')
        if prompts_path_str:
            prompts_path = Path(prompts_path_str)
            if not prompts_path.is_absolute():
                prompts_path = Path(__file__).resolve().parent.parent / prompts_path

    if prompts_path and prompts_path.exists():
        print(f"Extracting prompts from {prompts_path}...")
        prompts_by_id = extract_prompts_from_sft_jsonl(prompts_path)
        print(f"  Found {len(prompts_by_id)} prompts")
    else:
        if prompts_path:
            print(f"Warning: Prompts file not found: {prompts_path}", file=sys.stderr)
        print("Warning: No prompts provided; prompt column will be empty.", file=sys.stderr)

    # Initialize verifier
    print("Initializing verifier...")
    verifier = RuleBasedVerifier(config)

    # Process batch file
    stats = VerificationStats()
    results = process_batch_file(args.input, verifier, stats)

    # Print statistics
    print("\n" + "=" * 60)
    print(stats)
    print("=" * 60 + "\n")

    # Create and save SFT dataset (veRL-compatible parquet) to sft.dataset_dir
    if stats.passed > 0:
        sft_dataset_dir = config.get("sft", {}).get("dataset_dir", "data/sft_dataset")
        sft_output = Path(sft_dataset_dir) if Path(sft_dataset_dir).is_absolute() else Path(__file__).resolve().parent.parent / sft_dataset_dir
        print(f"Creating SFT parquet from {stats.passed} verified outputs...")
        dataset, saved_path = create_and_save_dataset(
            verified_results=results,
            output_dir=sft_output,
            prompts_by_id=prompts_by_id,
        )
        print(f"  SFT dataset saved to: {saved_path}")
        print(f"  Dataset size: {len(dataset)} examples (columns: {dataset.column_names})")

        # Save verification statistics to verification output dir
        stats_path = output_path / "verification_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                'total_processed': stats.total_processed,
                'passed': stats.passed,
                'failed': stats.failed,
                'pass_rate': stats.pass_rate,
                'failure_breakdown': stats.failure_breakdown,
            }, f, indent=2)
        print(f"  Statistics saved to: {stats_path}")
    else:
        print("Warning: No outputs passed verification. Dataset not created.", file=sys.stderr)
        sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()
