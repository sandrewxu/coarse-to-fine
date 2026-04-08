"""
Data utilities for step 5 (local latent generation).

Loading prompts, saving outputs, verification, and C2F flattening.
"""
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.data.schemas import VerificationStats
from src.verification import verify


def load_prompts(parquet_path: str | Path) -> list[str]:
    """Load the 'prompt' column from a parquet file."""
    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    return ds["prompt"]


def load_documents_from_jsonl(paths: list[str | Path]) -> list[str]:
    """Load raw documents from one or more JSONL/text files (e.g. chunk files).

    Each non-empty line is read.  If the line is valid JSON with a ``text``
    field, that field is used; otherwise the raw stripped line is used.
    """
    docs: list[str] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    docs.append(data.get("text", line))
                except json.JSONDecodeError:
                    docs.append(line)
    return docs


def resolve_chunk_paths(
    data_dir: str | Path,
    dataset_name: str,
    chunk_indices: list[int],
) -> list[Path]:
    """Build chunk file paths from dataset config and chunk indices."""
    data_dir = Path(data_dir)
    paths = []
    for i in chunk_indices:
        p = data_dir / f"{dataset_name}.chunk.{i:02d}.jsonl"
        paths.append(p)
    return paths


def save_generation_outputs(
    prompts: list[str],
    outputs: list[str],
    output_dir: Path,
    filename: str = "generations.parquet",
) -> Path:
    """Save raw generation results as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"generated_id": f"gen-{i:06d}", "prompt": p, "response": o}
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
    Run verification on generated outputs and filter to passing ones.

    Args:
        prompts: Input prompts (parallel with outputs).
        outputs: Raw generated texts (z_4:\\nz_3:... format).
        config: Experiment config (needs word_count_constraints, verification).

    Returns:
        Tuple of (filtered_prompts, filtered_outputs, stats).
    """
    word_count_constraints = config["word_count_constraints"]
    strict = config.get("verification", {}).get("strict_word_count", True)
    stats = VerificationStats()

    filtered_prompts = []
    filtered_outputs = []

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        result = verify(
            output, word_count_constraints,
            custom_id=f"gen-{i:06d}", strict_word_count=strict,
        )

        stats.total_processed += 1
        if result.passed:
            stats.record_pass()
            filtered_prompts.append(prompt)
            filtered_outputs.append(output)
        else:
            for reason in result.failure_reasons:
                stats.record_failure(reason)

    return filtered_prompts, filtered_outputs, stats


_LAYER_RE = re.compile(r"^z_\d+:\s*(.*)$")


def flatten_for_c2f(
    prompts: list[str],
    responses: list[str],
    output_dir: Path,
    filename: str = "c2f_train.parquet",
) -> Path:
    """
    Flatten verified outputs for C2F training.

    Strips z_n: labels, concatenates layer content words + prompt words into
    a single flat space-separated string. Word count constraints guarantee
    deterministic scale boundary recovery.

    Args:
        prompts: Original documents (text scale).
        responses: Verified raw responses (z_4:\\nz_3:... format).
        output_dir: Directory to save the parquet file.
        filename: Output filename.

    Returns:
        Path to saved C2F training parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt, response in zip(prompts, responses):
        layer_contents = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = _LAYER_RE.match(line)
            if match:
                layer_contents.append(match.group(1).strip())
        flat_text = " ".join(layer_contents + [prompt])
        records.append({"text": flat_text})

    ds = Dataset.from_list(records)
    path = output_dir / filename
    ds.to_parquet(str(path))
    return path
