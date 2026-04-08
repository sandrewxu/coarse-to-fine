import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import typer
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


app = typer.Typer(help="Compute char/token length statistics for SFT datasets with 'input' and 'output' fields.")


def _maybe_load_dataset(src: str, split: str) -> Dataset:
    """
    Load a dataset from:
    - a Hugging Face repo id (e.g., 'nband/sft_active_reading_v1'), or
    - a local directory saved via datasets.save_to_disk(...)
    """
    p = Path(src)
    if p.exists():
        try:
            ds_or_dict = load_from_disk(str(p))
            if isinstance(ds_or_dict, DatasetDict):
                return ds_or_dict[split]
            return ds_or_dict
        except Exception:
            # Fallback to load_dataset for local config-style datasets
            pass
    # Default: treat as HF repo or script
    return load_dataset(src, split=split)


def _chunk_iter(iterable: Iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _summarize(arr: List[int]) -> Dict[str, float]:
    if not arr:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "mean": 0,
            "std": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "total": 0,
        }
    a = np.asarray(arr, dtype=np.int64)
    return {
        "count": int(a.size),
        "min": int(a.min()),
        "p50": float(np.percentile(a, 50)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
        "total": int(a.sum()),
    }


def _compute_lengths_for_batch(
    inputs: List[str],
    outputs: List[str],
    tokenizer,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    # Char lengths
    in_chars = [len(s) for s in inputs]
    out_chars = [len(s) for s in outputs]
    comb_texts = [i + o for i, o in zip(inputs, outputs)]
    comb_chars = [len(s) for s in comb_texts]

    # Token lengths (exclude special tokens to reflect raw content length)
    in_tok = tokenizer(inputs, add_special_tokens=False, truncation=False)["input_ids"]
    out_tok = tokenizer(outputs, add_special_tokens=False, truncation=False)["input_ids"]
    comb_tok = tokenizer(comb_texts, add_special_tokens=False, truncation=False)["input_ids"]

    in_tokens = [len(x) for x in in_tok]
    out_tokens = [len(x) for x in out_tok]
    comb_tokens = [len(x) for x in comb_tok]

    return in_chars, out_chars, comb_chars, in_tokens, out_tokens, comb_tokens


@app.command()
def main(
    dataset: str = typer.Argument("nband/sft_active_reading_v1", help="HF repo id (e.g., 'nband/sft_active_reading_v1') or local dataset dir from datasets.save_to_disk."),
    tokenizer_dir: Path = typer.Argument("./out/checkpoints/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints/step-1440000/tokenizer", exists=True, dir_okay=True, file_okay=False, readable=True, help="Directory containing a HF tokenizer to load."),
    split: str = typer.Option("train", "--split", help="Dataset split to analyze."),
    input_col: str = typer.Option("input", "--input-col", help="Column name for input text."),
    output_col: str = typer.Option("output", "--output-col", help="Column name for output text."),
    batch_size: int = typer.Option(512, "--batch-size", min=1, help="Batch size for tokenization."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit number of samples for quick stats."),
    save_json: Optional[Path] = typer.Option(None, "--save-json", help="Optional path to save the stats as JSON."),
):
    # Load dataset
    ds = _maybe_load_dataset(dataset, split)

    # Validate columns
    for col in (input_col, output_col):
        if col not in ds.column_names:
            raise typer.BadParameter(f"Column '{col}' not found in dataset. Available: {ds.column_names}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

    in_chars_all: List[int] = []
    out_chars_all: List[int] = []
    comb_chars_all: List[int] = []
    in_tokens_all: List[int] = []
    out_tokens_all: List[int] = []
    comb_tokens_all: List[int] = []

    n = len(ds)
    limit = n if max_samples is None else min(n, int(max_samples))

    # Iterate in batches
    processed = 0
    for idx_chunk in _chunk_iter(range(limit), batch_size):
        inputs = [str(ds[i][input_col]) for i in idx_chunk]
        outputs = [str(ds[i][output_col]) for i in idx_chunk]

        ic, oc, cc, it, ot, ct = _compute_lengths_for_batch(inputs, outputs, tokenizer)

        in_chars_all.extend(ic)
        out_chars_all.extend(oc)
        comb_chars_all.extend(cc)
        in_tokens_all.extend(it)
        out_tokens_all.extend(ot)
        comb_tokens_all.extend(ct)

        processed += len(idx_chunk)

    # Summaries
    stats = {
        "dataset": dataset,
        "split": split,
        "num_samples": processed,
        "input": {
            "chars": _summarize(in_chars_all),
            "tokens": _summarize(in_tokens_all),
        },
        "output": {
            "chars": _summarize(out_chars_all),
            "tokens": _summarize(out_tokens_all),
        },
        "combined_input_plus_output": {
            "chars": _summarize(comb_chars_all),
            "tokens": _summarize(comb_tokens_all),
        },
    }

    # Pretty print
    def _fmt(d: Dict[str, float]) -> str:
        return (
            f"count={d['count']:,}  min={d['min']:,}  p50={d['p50']:.2f}  mean={d['mean']:.2f}  "
            f"std={d['std']:.2f}  p95={d['p95']:.2f}  p99={d['p99']:.2f}  max={d['max']:,}  total={d['total']:,}"
        )

    typer.secho(f"Dataset: {dataset}  split: {split}  samples: {processed:,}", fg=typer.colors.CYAN)
    typer.echo("")
    typer.secho("Input (chars):   " + _fmt(stats["input"]["chars"]), fg=typer.colors.GREEN)
    typer.secho("Input (tokens):  " + _fmt(stats["input"]["tokens"]), fg=typer.colors.GREEN)
    typer.echo("")
    typer.secho("Output (chars):  " + _fmt(stats["output"]["chars"]), fg=typer.colors.YELLOW)
    typer.secho("Output (tokens): " + _fmt(stats["output"]["tokens"]), fg=typer.colors.YELLOW)
    typer.echo("")
    typer.secho("Combined (chars):  " + _fmt(stats["combined_input_plus_output"]["chars"]), fg=typer.colors.MAGENTA)
    typer.secho("Combined (tokens): " + _fmt(stats["combined_input_plus_output"]["tokens"]), fg=typer.colors.MAGENTA)

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(stats, f, indent=2)
        typer.secho(f"\nSaved JSON stats to {save_json}", fg=typer.colors.BLUE)


if __name__ == "__main__":
    app()