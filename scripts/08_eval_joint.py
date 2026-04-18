#!/usr/bin/env python3
"""
Evaluate posterior collapse after joint ELBO training.

Loads the trained p_θ (C2F model) in causal mode and generates sequences
left-to-right from BOS.  If posterior collapse occurred, the z tokens
(z_4, z_3, z_2, z_1) should be degenerate / repetitive while the text
tokens remain reasonable.

Usage:
    python scripts/08_eval_joint.py --checkpoint checkpoints/rl/joint/c2f/step_100
    python scripts/08_eval_joint.py --checkpoint checkpoints/decoder --num-samples 20
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

from src.common.logging import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate C2F posterior collapse")
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="C2F model checkpoint directory"
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=None, help="Space tokenizer directory"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of sequences to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "latent_generation.yaml"
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        log.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    from src.config import load_config

    config = load_config(args.config)
    scale_lengths = config["scale_lengths"]

    # ── Load model ───────────────────────────────────────────────────────────
    from src.qwen3_joint.configuration import C2FConfig
    from src.qwen3_joint.modeling import C2FForCausalLM

    log.info(f"Loading C2F model from {args.checkpoint} (causal mode)...")
    model_config = C2FConfig.from_pretrained(str(args.checkpoint))
    model_config.mask_type = "causal"
    model = C2FForCausalLM(model_config)

    from src.rl.reward import _load_c2f_weights

    model = _load_c2f_weights(model, args.checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ── Load tokenizer ───────────────────────────────────────────────────────
    from src.c2f_training.tokenizer import load_or_train_space_tokenizer

    c2f_cfg = config.get("c2f_training", {})
    tokenizer_dir = args.tokenizer_dir or Path(
        c2f_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
    )
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = PROJECT_ROOT / tokenizer_dir

    tokenizer = load_or_train_space_tokenizer(
        tokenizer_dir=tokenizer_dir,
        data_dir=c2f_cfg.get("dataset_dir", "data/sft_dataset"),
        dataset_format=c2f_cfg.get("dataset_format", "sft"),
    )
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id

    # ── Generate ─────────────────────────────────────────────────────────────
    content_len = 1 + sum(scale_lengths)  # BOS + all scales
    log.info(f"Generating {args.num_samples} sequences ({content_len} tokens each)...\n")

    scale_names = ["z_4", "z_3", "z_2", "z_1", "text"]
    all_z_tokens: list[str] = []

    for i in range(args.num_samples):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=content_len - 1,  # BOS already provided
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0].cpu().tolist()

        # Split into scale segments
        log.info(f"--- Sample {i + 1} ---")
        pos = 1  # skip BOS
        for name, length in zip(scale_names, scale_lengths, strict=False):
            segment_ids = generated_ids[pos : pos + length]
            segment_text = tokenizer.decode(segment_ids, skip_special_tokens=True)
            log.info(f"  {name:>4s}: {segment_text}")
            if name != "text":
                all_z_tokens.extend(segment_text.split())
            pos += length
        log.info()

    # ── Collapse metrics ─────────────────────────────────────────────────────
    if all_z_tokens:
        total = len(all_z_tokens)
        unique = len(set(all_z_tokens))
        counter = Counter(all_z_tokens)
        top5 = counter.most_common(5)

        log.info("=" * 50)
        log.info("Collapse Metrics")
        log.info("=" * 50)
        log.info(f"  Total z tokens:  {total}")
        log.info(f"  Unique z tokens: {unique} ({unique / total:.1%})")
        log.info(f"  Top 5 z tokens:  {top5}")
        if unique / total < 0.1:
            log.info("  --> COLLAPSED: z tokens are highly degenerate")
        else:
            log.info("  --> NOT collapsed: z tokens show diversity")

    return 0


if __name__ == "__main__":
    sys.exit(main())
