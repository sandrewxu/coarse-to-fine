"""
Produce veRL-compatible SFT data from verified batch outputs.

Outputs parquet with exactly two columns: prompt (original text) and response
(raw z_n: format latent layers). The response preserves the z_4: ...\nz_3: ...
format so the SFT model learns to generate verifiable output.
"""
from pathlib import Path
from typing import Optional

from datasets import Dataset

from src.data.schemas import TrainingExample
from src.verification import VerificationResult


def create_training_dataset(
    verified_results: list[VerificationResult],
    prompts_by_id: Optional[dict[str, str]] = None,
) -> Dataset:
    """
    Create dataset from verified results with prompt and response columns.

    The response column preserves the raw z_n: format (e.g.,
    "z_4: Mia flower\\nz_3: Mia saw pretty flower\\n...") so the SFT model
    learns to generate output that can be verified by RuleBasedVerifier.

    Args:
        verified_results: List of verification results (only passed ones included)
        prompts_by_id: Mapping custom_id -> original text (prompt).
            If None or a key is missing, prompt is set to "".

    Returns:
        HuggingFace Dataset with columns prompt, response (veRL-compatible)
    """
    passed_results = [r for r in verified_results if r.passed]
    prompts_by_id = prompts_by_id or {}

    examples = []
    for result in passed_results:
        prompt = prompts_by_id.get(result.custom_id, "")
        response = result.raw_content
        example = TrainingExample(prompt=prompt, response=response)
        examples.append(example.to_dict())

    return Dataset.from_list(examples)


def save_training_dataset(
    dataset: Dataset,
    output_dir: str | Path,
    train_filename: str = "train.parquet",
) -> Path:
    """
    Save dataset as parquet for veRL (single file; no HF save_to_disk).

    Args:
        dataset: Dataset with prompt and response columns
        output_dir: Directory to save the parquet file
        train_filename: Parquet filename (default train.parquet)

    Returns:
        Path to the saved parquet file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / train_filename
    dataset.to_parquet(str(parquet_path))
    return parquet_path


def create_and_save_dataset(
    verified_results: list[VerificationResult],
    output_dir: str | Path,
    prompts_by_id: Optional[dict[str, str]] = None,
    train_filename: str = "train.parquet",
) -> tuple[Dataset, Path]:
    """
    Create and save SFT dataset in one step (veRL-compatible parquet only).

    Args:
        verified_results: List of verification results
        output_dir: Directory to save the parquet file (e.g. data/sft_dataset)
        prompts_by_id: custom_id -> prompt mapping for original text
        train_filename: Parquet filename for training data

    Returns:
        Tuple of (created dataset, path to saved parquet file)
    """
    dataset = create_training_dataset(verified_results, prompts_by_id)
    saved_path = save_training_dataset(dataset, output_dir, train_filename)
    return dataset, saved_path
