"""
Data loading and saving utilities for step 5 (local latent generation).

Handles loading prompts from the SFT parquet, saving raw generation outputs,
running verification, and flattening verified outputs for C2F training.
"""
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.data.schemas import BatchOutputItem, GenerationOutput, VerificationStats
from src.verification.rule_based import RuleBasedVerifier


def load_prompts(config: dict[str, Any]) -> list[str]:
    """
    Load prompt texts from the generation prompt dataset.

    Reads the 'prompt' column from the parquet specified in
    config['generation']['prompt_dataset'].

    Args:
        config: Full experiment config.

    Returns:
        List of prompt strings (original documents).
    """
    prompt_path = config["generation"]["prompt_dataset"]
    ds = load_dataset("parquet", data_files=prompt_path, split="train")
    return ds["prompt"]


def save_generation_outputs(
    prompts: list[str],
    outputs: list[str],
    output_dir: Path,
    filename: str = "generations.parquet",
) -> Path:
    """
    Save raw generation results as parquet.

    Args:
        prompts: Input prompts (original documents).
        outputs: Raw generated texts (z_4: ...\nz_3: ... format).
        output_dir: Directory to save the parquet file.
        filename: Output filename.

    Returns:
        Path to saved parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        GenerationOutput(
            generated_id=f"gen-{i:06d}",
            prompt=p,
            response=o,
        ).model_dump()
        for i, (p, o) in enumerate(zip(prompts, outputs))
    ]
    ds = Dataset.from_list(records)
    path = output_dir / filename
    ds.to_parquet(str(path))
    return path


def verify_and_filter_outputs(
    prompts: list[str],
    outputs: list[str],
    config: dict[str, Any],
) -> tuple[list[str], list[str], VerificationStats]:
    """
    Run rule-based verification on generated outputs and filter to passing ones.

    Reuses RuleBasedVerifier from src/verification/rule_based.py. The generated
    outputs are in the same z_4:\nz_3:... format as batch API output.

    Args:
        prompts: Input prompts (parallel with outputs).
        outputs: Raw generated texts.
        config: Full experiment config (needs word_count_constraints, verification).

    Returns:
        Tuple of (filtered_prompts, filtered_outputs, stats).
    """
    verifier = RuleBasedVerifier(config)
    stats = VerificationStats()

    filtered_prompts = []
    filtered_outputs = []

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        item = BatchOutputItem(
            custom_id=f"gen-{i:06d}",
            content=output,
            status_code=200,
        )
        result = verifier.verify(item)

        stats.total_processed += 1
        if result.passed:
            stats.record_pass()
            filtered_prompts.append(prompt)
            filtered_outputs.append(output)
        else:
            for reason in result.failure_reasons:
                stats.record_failure(reason)

    return filtered_prompts, filtered_outputs, stats


def _strip_layer_labels(response: str) -> list[str]:
    """
    Parse z_n: labels from a response and return content strings per layer.

    Args:
        response: Raw response with z_4: ...\nz_3: ... format.

    Returns:
        List of content strings (one per layer, in order).
    """
    layer_pattern = re.compile(r"^z_\d+:\s*(.*)$")
    contents = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = layer_pattern.match(line)
        if match:
            contents.append(match.group(1).strip())
    return contents


def flatten_for_c2f(
    prompts: list[str],
    responses: list[str],
    output_dir: Path,
    filename: str = "c2f_train.parquet",
) -> Path:
    """
    Flatten verified outputs for C2F training.

    Strips z_n: labels, concatenates layer content words + prompt words into
    a single flat space-separated string. The word count constraints guarantee
    deterministic scale boundary recovery:
      words 0-1 = z_4 (2 words)
      words 2-5 = z_3 (4 words)
      words 6-13 = z_2 (8 words)
      words 14-29 = z_1 (16 words)
      words 30+ = text (32 words)

    Args:
        prompts: Original documents (text scale).
        responses: Verified raw responses (z_4:\nz_3:... format).
        output_dir: Directory to save the parquet file.
        filename: Output filename.

    Returns:
        Path to saved C2F training parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt, response in zip(prompts, responses):
        layer_contents = _strip_layer_labels(response)
        # Concatenate: latent layer words + prompt words
        flat_parts = layer_contents + [prompt]
        flat_text = " ".join(flat_parts)
        records.append({"text": flat_text})

    ds = Dataset.from_list(records)
    path = output_dir / filename
    ds.to_parquet(str(path))
    return path
