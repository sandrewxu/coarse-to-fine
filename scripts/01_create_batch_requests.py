#!/usr/bin/env python3
"""
Create OpenAI Batch API request JSONL files for latent generation.

Usage:
    # Derive docs path from config (dataset.data_dir / dataset.prompt_split)
    python scripts/01_create_batch_requests.py \
        --config config/latent_generation.yaml

    # Explicit docs path
    python scripts/01_create_batch_requests.py \
        --docs data/tinystoriesv2_shuffled/tinystoriesv2.prompt.jsonl \
        --config config/latent_generation.yaml

    # Override batch params
    python scripts/01_create_batch_requests.py \
        --config config/latent_generation.yaml \
        --doc-start -10000 \
        --model gpt-5-nano-2025-08-07
"""

import argparse
import random
import sys
from pathlib import Path

from src.batch.requests import (
    create_batch_requests,
    load_documents,
    load_few_shot_examples,
    save_batch_requests,
)
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT
from src.config import load_config

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create OpenAI Batch API request JSONL files")
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Document JSONL/text file (default: derived from dataset config)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment config YAML")
    parser.add_argument(
        "--doc-start", type=int, default=0, help="Start document index (supports negative)"
    )
    parser.add_argument("--doc-end", type=int, default=None, help="End document index")
    parser.add_argument("--model", type=str, default=None, help="Batch API model")
    parser.add_argument("--reasoning-effort", type=str, default=None, help="Reasoning effort level")
    parser.add_argument("--verbosity", type=str, default=None, help="Verbosity level")
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="System prompt name (without .txt)"
    )
    parser.add_argument(
        "--user-prompt", type=str, default=None, help="User prompt name (without .txt)"
    )
    parser.add_argument(
        "--few-shot", type=str, default=None, help="Few-shot examples name (without .jsonl)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for request JSONL"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Load config (if provided)
    batch_config: dict = {}
    dataset_config: dict = {}
    if args.config:
        config = load_config(args.config)
        batch_config = config.get("batch", {})
        dataset_config = config.get("dataset", {})

    # Resolve docs path: CLI > config (dataset.data_dir / dataset.prompt_split)
    docs_path = args.docs
    if docs_path is None:
        data_dir = dataset_config.get("data_dir", "data/tinystoriesv2_shuffled")
        prompt_split = dataset_config.get("prompt_split", "tinystoriesv2.prompt.jsonl")
        docs_path = Path(data_dir) / prompt_split
        if not docs_path.is_absolute():
            docs_path = PROJECT_ROOT / docs_path

    if not docs_path.exists():
        log.error(f"Document file not found: {docs_path}")
        sys.exit(1)

    # CLI overrides config
    model = args.model or batch_config.get("model", "gpt-5-nano-2025-08-07")
    reasoning_effort = args.reasoning_effort or batch_config.get("reasoning_effort", "low")
    verbosity = args.verbosity or batch_config.get("verbosity", "medium")
    system_prompt_name = args.system_prompt or batch_config.get(
        "system_prompt", "latent-generation"
    )
    user_prompt_name = args.user_prompt or batch_config.get("user_prompt", "gemini-3-pro-7")
    few_shot_name = args.few_shot or batch_config.get("few_shot_examples", "latent-generation")
    output_dir = args.output_dir or Path(batch_config.get("prompt_data_dir", "data/prompt_data"))

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    random.seed(args.seed)

    # Load documents
    documents = load_documents(docs_path, start=args.doc_start, end=args.doc_end)
    log.error(
        f"Loaded {len(documents)} documents from {docs_path} [{args.doc_start}:{args.doc_end or 'end'}]"
    )

    # Load prompt templates
    prompts_dir = PROJECT_ROOT / "prompts"

    system_prompt_path = prompts_dir / "system_prompts" / f"{system_prompt_name}.txt"
    if not system_prompt_path.exists():
        log.error(f"System prompt not found: {system_prompt_path}")
        sys.exit(1)
    system_prompt = system_prompt_path.read_text().strip()

    user_prompt_path = prompts_dir / "user_prompts" / f"{user_prompt_name}.txt"
    if not user_prompt_path.exists():
        log.error(f"User prompt not found: {user_prompt_path}")
        sys.exit(1)
    user_prompt_template = user_prompt_path.read_text().strip()

    few_shot_messages: list[dict] = []
    if few_shot_name:
        few_shot_path = prompts_dir / "few_shot_examples" / f"{few_shot_name}.jsonl"
        if few_shot_path.exists():
            few_shot_messages = load_few_shot_examples(few_shot_path)
            log.error(f"Loaded {len(few_shot_messages) // 2} few-shot examples")
        else:
            log.warning(f"Few-shot file not found: {few_shot_path}")

    # Create requests
    requests = create_batch_requests(
        documents=documents,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        few_shot_messages=few_shot_messages,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    # Save with organized directory structure
    output_path = (
        output_dir
        / model
        / f"docs_{len(documents):010d}"
        / user_prompt_name
        / system_prompt_name
        / "sft.jsonl"
    )
    save_batch_requests(requests, output_path)
    log.info(f"Saved {len(requests)} requests to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
