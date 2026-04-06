#!/usr/bin/env python3
"""
Generate latent outputs at scale using vLLM from the SFT-finetuned model.

Reads prompts from the SFT dataset (or a custom prompt source), generates latent
layers conditioned on each prompt, verifies outputs, and produces both a raw
generation parquet and a flattened C2F training parquet.

Usage:
    python scripts/05_generate_local.py \
        --config config/latent_generation.yaml

    # Override number of samples or output directory:
    python scripts/05_generate_local.py \
        --config config/latent_generation.yaml \
        --num-samples 1000 \
        --output-dir data/local_generations/run2

Requires: pip install -e ".[generation]" and a CUDA-capable environment.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate latent outputs from SFT model via vLLM"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "latent_generation.yaml",
        help="Path to experiment YAML with 'generation' section",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override: number of prompts to generate for (default: all prompts in dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override: output directory for generation results",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config not found: {args.config}", file=sys.stderr)
        return 1

    # Load .env (secrets) and configure W&B before any training imports
    from src.config import load_config
    from src.utils.env import load_env, setup_wandb

    load_env()
    config = load_config(args.config)
    wandb_enabled = setup_wandb(config, step_name="generation")
    if wandb_enabled:
        print("W&B logging enabled for generation")

    if "generation" not in config:
        print("Error: Config missing 'generation' section", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.output_dir:
        config["generation"]["output_dir"] = str(args.output_dir)

    # Resolve output directory
    output_dir = Path(config["generation"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Initialize W&B run (generation is not HF Trainer, so we init manually)
    wandb_run = None
    if wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            name=f"generation-{config['generation'].get('seed', 42)}",
            config={
                "step": "05-generate-local",
                "generation": config["generation"],
                "scale_lengths": config.get("scale_lengths"),
                "word_count_constraints": config.get("word_count_constraints"),
            },
        )

    # 1. Load prompts
    from src.generation.dataset import (
        flatten_for_c2f,
        load_prompts,
        save_generation_outputs,
        verify_and_filter_outputs,
    )

    print("Loading prompts...")
    prompts = load_prompts(config)

    if args.num_samples and args.num_samples < len(prompts):
        prompts = prompts[: args.num_samples]

    print(f"  Loaded {len(prompts)} prompts")

    # 2. Generate
    from src.generation.inference import LatentGenerator

    print("Initializing vLLM engine...")
    generator = LatentGenerator(config)

    print(f"Generating latent outputs for {len(prompts)} prompts...")
    outputs = generator.generate(prompts)
    print(f"  Generated {len(outputs)} outputs")

    # 3. Save raw outputs
    raw_path = save_generation_outputs(prompts, outputs, output_dir)
    print(f"  Raw generations saved to: {raw_path}")

    total_generated = len(outputs)

    # 4. Verify and filter
    if config["generation"].get("verify_outputs", True):
        print("Verifying outputs...")
        prompts, outputs, stats = verify_and_filter_outputs(prompts, outputs, config)
        print(f"\n{stats}\n")
        print(f"  {len(outputs)} outputs passed verification")

        # Log verification metrics to W&B
        if wandb_run is not None:
            wandb_run.log({
                "generation/total_generated": total_generated,
                "generation/total_passed": len(outputs),
                "generation/pass_rate": len(outputs) / max(total_generated, 1),
                "generation/total_failed": total_generated - len(outputs),
            })
    else:
        print("  Skipping verification (verify_outputs=false)")

    if not outputs:
        print("Error: No outputs passed verification.", file=sys.stderr)
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        return 1

    # 5. Flatten for C2F training
    print("Flattening verified outputs for C2F training...")
    c2f_path = flatten_for_c2f(prompts, outputs, output_dir)
    print(f"  C2F training data saved to: {c2f_path}")

    # Log final dataset size
    if wandb_run is not None:
        wandb_run.log({"generation/c2f_training_samples": len(outputs)})
        wandb_run.finish()

    print(f"\nDone! {len(outputs)} verified samples ready for C2F training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
