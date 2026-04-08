"""Dataset metadata registry for supported HuggingFace datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a downloadable dataset."""

    repo_id: str
    allow_patterns: str | None = None
    orig_extension: str = ".jsonl"
    cat_command: str = "cat {}"
    needs_parquet_conversion: bool = False
    needs_preprocessing: bool = False


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "fineweb_edu": DatasetInfo(
        repo_id="HuggingFaceFW/fineweb-edu",
        needs_parquet_conversion=True,
    ),
    "fineweb_edu_10bt": DatasetInfo(
        repo_id="HuggingFaceFW/fineweb-edu",
        allow_patterns="sample/10BT/*",
        needs_parquet_conversion=True,
    ),
    "dclm_baseline_1.0": DatasetInfo(
        repo_id="mlfoundations/dclm-baseline-1.0",
        allow_patterns="*.jsonl.zst",
        orig_extension=".jsonl.zst",
        cat_command="zstdcat {} && echo",
    ),
    "dclm_baseline_1.0_10prct": DatasetInfo(
        repo_id="mlfoundations/dclm-baseline-1.0",
        allow_patterns="global-shard_01_of_10/*.jsonl.zst",
        orig_extension=".jsonl.zst",
        cat_command="zstdcat {} && echo",
    ),
    "dclm_pool_1b_1x": DatasetInfo(
        repo_id="mlfoundations/dclm-pool-1b-1x",
        allow_patterns="*.jsonl.zst",
        orig_extension=".jsonl.zst",
        cat_command="zstdcat {} && echo",
    ),
    "cosmopedia_v2": DatasetInfo(
        repo_id="HuggingFaceTB/smollm-corpus",
        allow_patterns="cosmopedia-v2/*",
        needs_parquet_conversion=True,
    ),
    "python_edu": DatasetInfo(
        repo_id="HuggingFaceTB/smollm-corpus",
        allow_patterns="python-edu/*",
    ),
    "tinystoriesv2": DatasetInfo(
        repo_id="roneneldan/TinyStories",
        allow_patterns="TinyStoriesV2-GPT4-*.txt",
        orig_extension=".txt",
        needs_preprocessing=True,
    ),
}
