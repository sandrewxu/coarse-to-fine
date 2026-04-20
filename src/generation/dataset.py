"""
Data utilities for step 5 (local latent generation).

Loading prompts, saving outputs, verification, and C2F flattening. Also
provides ``build_rl_parquet`` (used by the RL phase orchestration in step 7
to convert the RL-split JSONL into veRL's expected parquet schema).
"""

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.common.logging import get_logger
from src.data.schemas import VerificationStats
from src.verification import verify

log = get_logger(__name__)


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
                    if isinstance(data, dict):
                        docs.append(data.get("text", line))
                    else:
                        docs.append(str(data))
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
        for i, (p, o) in enumerate(zip(prompts, outputs, strict=False))
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

    for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=False)):
        result = verify(
            output,
            word_count_constraints,
            custom_id=f"gen-{i:06d}",
            strict_word_count=strict,
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
    for prompt, response in zip(prompts, responses, strict=False):
        layer_contents = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = _LAYER_RE.match(line)
            if match:
                layer_contents.append(match.group(1).strip())
        flat_text = " ".join([*layer_contents, prompt])
        records.append({"text": flat_text})

    ds = Dataset.from_list(records)
    path = output_dir / filename
    ds.to_parquet(str(path))
    return path


def build_rl_parquet(
    config: dict[str, Any],
    project_root: Path,
    *,
    rl_section: str = "sft_rl",
    split: str = "rl",
) -> Path:
    """Build an RL parquet (train or val) from a JSONL document split.

    Reads raw documents from ``dataset.rl_split`` (or ``dataset.val_split`` when
    ``split="val"``) and creates a veRL-compatible parquet with columns:

    - ``prompt``: the raw document text wrapped as a chat message (input to
      ``q_φ`` during rollout).
    - ``ground_truth``: copy of the document (used by the reward manager to run
      the C2F forward pass).
    - ``data_source``: literal ``"latent_generation"``.
    - ``is_validation``: bool flag. False for train, True for val. Read by the
      reward manager to skip p_θ updates on validation samples (veRL's
      ``agent_loop._compute_score`` drops ``meta_info``, so the ``validate``
      flag can't reach ``run_single`` any other way).

    Args:
        config: Full experiment config.
        project_root: Repo root, used to resolve relative paths in the config.
        rl_section: Which RL subsection holds the dataset_dir (``"sft_rl"`` or
            ``"joint"``).
        split: ``"rl"`` for the training split (reads ``dataset.rl_split``,
            writes ``sft_rl.parquet``); ``"val"`` for the validation split
            (reads ``dataset.val_split``, writes ``sft_rl_val.parquet``).

    Returns:
        Path to the written parquet. If a parquet already exists at the target
        path, it is kept and returned as-is (idempotent).
    """
    if split not in ("rl", "val"):
        raise ValueError(f"split must be 'rl' or 'val', got {split!r}")
    is_validation = split == "val"

    rl_cfg = config["rl"][rl_section]

    rl_dataset_dir = Path(rl_cfg.get("dataset_dir", "data/rl_dataset"))
    if not rl_dataset_dir.is_absolute():
        rl_dataset_dir = project_root / rl_dataset_dir
    rl_dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = "sft_rl_val.parquet" if is_validation else "sft_rl.parquet"
    rl_parquet = rl_dataset_dir / filename
    if rl_parquet.exists():
        log.info("RL parquet already exists: %s", rl_parquet)
        return rl_parquet

    dataset_cfg = config.get("dataset", {})
    data_dir = Path(dataset_cfg.get("data_dir", "data/tinystoriesv2_shuffled"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    split_key = "val_split" if is_validation else "rl_split"
    default_filename = "tinystoriesv2.val.jsonl" if is_validation else "tinystoriesv2.rl.jsonl"
    split_file = data_dir / dataset_cfg.get(split_key, default_filename)

    if not split_file.exists():
        raise FileNotFoundError(
            f"{split!r} split not found: {split_file}\nRun step 0 (00_prepare_data.py) first."
        )

    log.info("Preparing %s parquet from %s ...", split, split_file)
    docs = load_documents_from_jsonl([split_file])

    # ``reward_model.ground_truth`` is the shape veRL's NaiveRewardManager
    # (and other built-in managers) expect; our C2F managers fall back to the
    # flat ``ground_truth`` key. Write both so the parquet works regardless of
    # which reward manager ends up loaded.
    records = [
        {
            "prompt": [{"role": "user", "content": doc}],
            "ground_truth": doc,
            "reward_model": {"ground_truth": doc},
            "data_source": "latent_generation",
            "is_validation": is_validation,
        }
        for doc in docs
    ]
    ds = Dataset.from_list(records)
    ds.to_parquet(str(rl_parquet))
    log.info("Saved %s parquet: %s (%d samples)", split, rl_parquet, len(ds))
    return rl_parquet
