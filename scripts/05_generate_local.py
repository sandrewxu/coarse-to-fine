#!/usr/bin/env python3
"""
Generate latent outputs from the SFT-finetuned model.

Usage:
    python scripts/05_generate_local.py \
        --data data/sft_dataset/train.parquet \
        --model checkpoints/sft/global_step_292/huggingface \
        --output-dir data/local_generations

    # Use HuggingFace backend instead of vLLM:
    python scripts/05_generate_local.py \
        --data data/sft_dataset/train.parquet \
        --model checkpoints/sft/global_step_292/huggingface \
        --output-dir data/local_generations \
        --backend hf

    # With config for sampling params, verification, and W&B:
    python scripts/05_generate_local.py \
        --data data/sft_dataset/train.parquet \
        --model checkpoints/sft/global_step_292/huggingface \
        --output-dir data/local_generations \
        --config config/latent_generation.yaml \
        --num-samples 1000
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate latent outputs from SFT model")
    parser.add_argument("--data", required=True, type=Path, help="Parquet file with 'prompt' column")
    parser.add_argument("--model", required=True, type=str, help="SFT model checkpoint path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"], help="Generation backend (default: vllm)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (vLLM tensor parallel)")
    parser.add_argument("--num-samples", type=int, default=None, help="Generate for first N prompts only")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML for sampling params, verification, W&B")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}", file=sys.stderr)
        return 1

    # Load config for defaults (or use empty)
    config = {}
    gen_config = {}
    wandb_enabled = False

    if args.config:
        from src.config import load_config
        from src.utils.env import load_env, setup_wandb
        load_env()
        config = load_config(args.config)
        gen_config = config.get("generation", {})
        wandb_enabled = setup_wandb(config, step_name="generation")

    # Load prompts
    from src.generation.dataset import load_prompts
    print(f"Loading prompts from {args.data}...")
    prompts = load_prompts(args.data)
    if args.num_samples and args.num_samples < len(prompts):
        prompts = prompts[:args.num_samples]
    print(f"  {len(prompts)} prompts loaded")

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
    print(f"Generating with {args.backend} backend ({len(prompts)} prompts)...")
    outputs = generate(args.backend, args.model, prompts, **sampling_kwargs)
    print(f"  Generated {len(outputs)} outputs")

    # Save raw outputs
    from src.generation.dataset import save_generation_outputs
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    raw_path = save_generation_outputs(prompts, outputs, output_dir)
    print(f"  Raw generations saved to: {raw_path}")

    # Verify and flatten (if config provided with word_count_constraints)
    if config.get("word_count_constraints"):
        from src.generation.dataset import flatten_for_c2f, verify_and_filter_outputs

        if gen_config.get("verify_outputs", True):
            print("Verifying outputs...")
            prompts, outputs, stats = verify_and_filter_outputs(prompts, outputs, config)
            print(f"\n{stats}\n")

            if not outputs:
                print("Error: No outputs passed verification.", file=sys.stderr)
                return 1

        print("Flattening for C2F training...")
        c2f_path = flatten_for_c2f(prompts, outputs, output_dir)
        print(f"  C2F training data saved to: {c2f_path}")

    print(f"\nDone! {len(outputs)} samples in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
