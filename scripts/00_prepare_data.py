#!/usr/bin/env python3
"""
Download, preprocess, shuffle, and split a HuggingFace dataset.

Produces sharded JSONL chunks plus dedicated validation, test, prompt,
and RL splits ready for the rest of the pipeline.

Usage:
    python scripts/00_prepare_data.py --dataset tinystoriesv2 --memory 8
    python scripts/00_prepare_data.py --dataset tinystoriesv2 --config config/latent_generation.yaml
    python scripts/00_prepare_data.py --dataset fineweb_edu_10bt --memory 32 --data-dir /scratch/data
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_tinystories
from src.data.registry import DATASET_REGISTRY


# ---------------------------------------------------------------------------
# Download & tool setup  (kept in-script — infrastructure, not reusable lib)
# ---------------------------------------------------------------------------

def run_command(command: str) -> None:
    """Run a shell command, printing it first."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id: str, local_dir: str, allow_patterns: str | None) -> None:
    """Download a dataset from HuggingFace Hub with retries."""
    from huggingface_hub import snapshot_download

    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,
            )
            break
        except Exception:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def setup_terashuf(work_dir: str) -> str:
    """Clone and build terashuf if not already present. Returns terashuf dir."""
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def parquet_to_jsonl(dataset: str, work_dir: str, src_dir: str, tgt_dir: str, ntasks: int = 64) -> None:
    """Convert parquet files to JSONL using datatrove."""
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(src_dir, file_progress=True, doc_progress=True, glob_pattern="**/*.parquet"),
            JsonlWriter(tgt_dir, output_filename=dataset + ".chunk.${rank}.jsonl", compression=None),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download, preprocess, shuffle, and split a dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. tinystoriesv2)")
    parser.add_argument("--config", type=Path, default=None, help="Experiment config YAML")
    parser.add_argument("--memory", type=float, default=None, help="Terashuf memory in GB")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--final-data-dir", type=str, default=None, help="Final destination for shuffled data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--nchunks", type=int, default=None, help="Number of shards")
    parser.add_argument("--clear-work-dir", action="store_true", help="Clear work dir after transfer")
    args = parser.parse_args()

    # Load config defaults
    prep_config: dict = {}
    if args.config:
        from src.config import load_config
        config = load_config(args.config)
        prep_config = config.get("data_prep", {})

    # CLI overrides config
    dataset = args.dataset or prep_config.get("dataset", "tinystoriesv2")
    memory = args.memory or prep_config.get("memory_gb", 8.0)
    data_dir = args.data_dir or prep_config.get("raw_data_dir", "data")
    seed = args.seed or prep_config.get("seed", 42)
    nchunks = args.nchunks or prep_config.get("num_chunks", 8)
    final_data_dir = args.final_data_dir
    clear_work_dir = args.clear_work_dir

    k_validation = prep_config.get("k_validation", 2500)
    k_test = prep_config.get("k_test", 2500)
    k_prompt = prep_config.get("k_prompt", 12500)
    k_rl = prep_config.get("k_rl", 4000)
    words_per_chunk = prep_config.get("words_per_chunk", 32)

    # Validate dataset
    if dataset not in DATASET_REGISTRY:
        print(f"Error: Unknown dataset '{dataset}'. Available: {list(DATASET_REGISTRY.keys())}", file=sys.stderr)
        return 1

    info = DATASET_REGISTRY[dataset]

    # Paths
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir
    prefix = f"{dataset}.chunk."
    suffix = ".jsonl"

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download
    download_dataset(info.repo_id, src_dir, info.allow_patterns)

    # Format conversion (parquet → JSONL)
    if info.needs_parquet_conversion:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # TinyStories preprocessing
    if info.needs_preprocessing:
        preprocess_tinystories(src_dir, words_per_chunk=words_per_chunk)

    # Environment for terashuf
    os.environ["MEMORY"] = str(memory)
    os.environ["SEED"] = str(seed)
    os.environ["TMPDIR"] = os.environ.get("TMPDIR", "/tmp")

    # Shuffle and shard
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{info.orig_extension}' -print0 | "
        f"xargs -0 -I {{}} sh -c '{info.cat_command}' | "
        f"{terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length=2 --additional-suffix={suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' PIPE;"
    )

    # Create validation, test, prompt, and RL splits from chunks
    val_file = f"{out_dir}/{dataset}.val{suffix}"
    test_file = f"{out_dir}/{dataset}.test{suffix}"
    prompt_file = f"{out_dir}/{dataset}.prompt{suffix}"
    rl_file = f"{out_dir}/{dataset}.rl{suffix}"

    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_test} {chunk_file} >> {test_file}")
        run_command(f"sed -i '1,{k_test}d' {chunk_file}")
        run_command(f"head -n {k_validation} {chunk_file} >> {val_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")
        run_command(f"head -n {k_prompt} {chunk_file} >> {prompt_file}")
        run_command(f"sed -i '1,{k_prompt}d' {chunk_file}")
        run_command(f"head -n {k_rl} {chunk_file} >> {rl_file}")
        run_command(f"sed -i '1,{k_rl}d' {chunk_file}")

    print(f"Processing completed in {work_dir}")

    # Optional: transfer to final location
    if final_data_dir:
        final_out_dir = os.path.join(final_data_dir, f"{dataset}_shuffled")
        print(f"Copying shuffled data to: {final_out_dir}")
        os.makedirs(final_data_dir, exist_ok=True)
        run_command(f"rsync -avP --delete {out_dir}/ {final_out_dir}")

        if clear_work_dir:
            print("Clearing work directories...")
            run_command(f"rm -rf {src_dir}")
            run_command(f"rm -rf {out_dir}")

    print("All tasks completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
