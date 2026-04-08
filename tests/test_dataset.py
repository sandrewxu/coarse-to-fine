"""Tests for dataset utilities."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq


def _detect_format(parquet_path: Path) -> str:
    """Replicate the detection logic from 06_train_decoder.py."""
    columns = pq.read_schema(str(parquet_path)).names
    if "text" in columns:
        return "c2f"
    if "prompt" in columns and "response" in columns:
        return "sft"
    raise ValueError(f"Unknown format: {columns}")


def test_detect_sft_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        table = pa.table({"prompt": ["hello"], "response": ["z_4: a b"]})
        pq.write_table(table, str(path))
        assert _detect_format(path) == "sft"


def test_detect_c2f_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        table = pa.table({"text": ["hello world"]})
        pq.write_table(table, str(path))
        assert _detect_format(path) == "c2f"


def test_sft_dataset_creation():
    """Test creating SFT dataset from verified results."""
    from src.verification import VerificationResult
    from src.sft.dataset import create_training_dataset

    results = [
        VerificationResult(custom_id="test-1", passed=True, raw_content="z_4: a b\nz_3: c d e f"),
        VerificationResult(custom_id="test-2", passed=False, raw_content="bad"),
        VerificationResult(custom_id="test-3", passed=True, raw_content="z_4: g h\nz_3: i j k l"),
    ]
    prompts = {"test-1": "original text one", "test-3": "original text three"}

    dataset = create_training_dataset(results, prompts)
    assert len(dataset) == 2  # only passed results
    assert dataset[0]["prompt"] == "original text one"
    assert dataset[0]["response"] == "z_4: a b\nz_3: c d e f"
