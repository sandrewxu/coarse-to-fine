#!/usr/bin/env python3
"""
Generate latent outputs from the SFT-finetuned model.

Usage:
    # Generate from chunk files (raw documents):
    python scripts/05_generate_local.py \
        --chunks 0 1 2 3 \
        --model checkpoints/sft/global_step_292/huggingface \
        --config config/latent_generation.yaml

    # Generate from a parquet file (backward compat):
    python scripts/05_generate_local.py \
        --data data/sft_dataset/train.parquet \
        --model checkpoints/sft/global_step_292/huggingface \
        --output-dir data/local_generations

    # Subset of samples:
    python scripts/05_generate_local.py \
        --chunks 0 \
        --model checkpoints/sft/global_step_292/huggingface \
        --config config/latent_generation.yaml \
        --num-samples 1000
"""

import argparse
import sys
from pathlib import Path

from src.common.logging import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate latent outputs from SFT model")
    # Data source: --chunks (from dataset config) or --data (explicit parquet)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=None,
        help="Chunk indices to generate from (e.g. 0 1 2 3)",
    )
    data_group.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Parquet file with 'prompt' column (backward compat)",
    )
    parser.add_argument("--model", type=str, default=None, help="SFT model checkpoint path")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "hf"],
        help="Generation backend (default: vllm)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs (vLLM tensor parallel)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Generate for first N prompts only"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML for sampling params, verification, W&B",
    )
    args = parser.parse_args()

    # Load config for defaults (or use empty)
    config = {}
    gen_config = {}
    dataset_config = {}

    if args.config:
        from src.common.env import load_env, setup_wandb
        from src.config import load_config

        load_env()
        config = load_config(args.config)
        gen_config = config.get("generation", {})
        dataset_config = config.get("dataset", {})
        setup_wandb(config, step_name="generation")

    # Resolve model path
    model_path = args.model or gen_config.get("model_path", "")
    if not model_path:
        log.error("--model is required (or set generation.model_path in config)")
        return 1

    # Resolve output dir
    output_dir = args.output_dir or Path(gen_config.get("output_dir", "data/local_generations"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Load prompts from chunks or parquet
    if args.data is not None:
        # Backward compat: load from parquet
        if not args.data.exists():
            log.error(f"Data file not found: {args.data}")
            return 1
        from src.generation.dataset import load_prompts

        log.error(f"Loading prompts from {args.data}...")
        prompts = load_prompts(args.data)
    else:
        # Load from chunk files
        chunk_indices = args.chunks or gen_config.get("chunks", [0, 1, 2, 3])
        data_dir = dataset_config.get("data_dir", "data/tinystoriesv2_shuffled")
        dataset_name = dataset_config.get("dataset_name", "tinystoriesv2")
        if not Path(data_dir).is_absolute():
            data_dir = str(PROJECT_ROOT / data_dir)

        from src.generation.dataset import load_documents_from_jsonl, resolve_chunk_paths

        chunk_paths = resolve_chunk_paths(data_dir, dataset_name, chunk_indices)

        # Verify all chunk files exist
        for p in chunk_paths:
            if not p.exists():
                log.error(f"Chunk file not found: {p}")
                return 1

        log.error(f"Loading documents from {len(chunk_paths)} chunks: {chunk_indices}...")
        prompts = load_documents_from_jsonl(chunk_paths)

    if args.num_samples and args.num_samples < len(prompts):
        prompts = prompts[: args.num_samples]
    log.info(f"  {len(prompts)} prompts loaded")

    # Build sampling kwargs from config defaults, CLI overrides take priority
    sampling_kwargs = {
        "temperature": gen_config.get("temperature", 0.7),
        "top_p": gen_config.get("top_p", 0.9),
        "repetition_penalty": gen_config.get("repetition_penalty", 1.0),
    }
    if args.backend == "vllm":
        sampling_kwargs["max_tokens"] = gen_config.get("max_tokens", 256)
        sampling_kwargs["top_k"] = gen_config.get("top_k", -1)
        sampling_kwargs["num_gpus"] = args.num_gpus
        sampling_kwargs["seed"] = gen_config.get("seed", 42)
    else:
        sampling_kwargs["max_new_tokens"] = gen_config.get("max_tokens", 256)
        sampling_kwargs["top_k"] = gen_config.get("top_k", 50)

    # Generate
    from src.generation.inference import generate

    log.info(f"Generating with {args.backend} backend ({len(prompts)} prompts)...")
    outputs = generate(args.backend, model_path, prompts, **sampling_kwargs)
    log.info(f"  Generated {len(outputs)} outputs")

    # Save raw outputs
    from src.generation.dataset import save_generation_outputs

    raw_path = save_generation_outputs(prompts, outputs, output_dir)
    log.info(f"  Raw generations saved to: {raw_path}")

    # Verify and flatten (if config provided with word_count_constraints)
    if config.get("word_count_constraints"):
        from src.generation.dataset import flatten_for_c2f, verify_and_filter_outputs

        if gen_config.get("verify_outputs", True):
            log.info("Verifying outputs...")
            prompts, outputs, stats = verify_and_filter_outputs(prompts, outputs, config)
            log.info(f"\n{stats}\n")

            if not outputs:
                log.error("No outputs passed verification.")
                return 1

        log.info("Flattening for C2F training...")
        c2f_path = flatten_for_c2f(prompts, outputs, output_dir)
        log.info(f"  C2F training data saved to: {c2f_path}")

    log.info(f"\nDone! {len(outputs)} samples in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
