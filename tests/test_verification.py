"""Tests for space-based verification logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification import verify

CONSTRAINTS = {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}


def _make_content(**overrides):
    """Build valid z_N content, with optional per-layer word count overrides."""
    counts = {**CONSTRAINTS, **overrides}
    lines = []
    for name in ["z_4", "z_3", "z_2", "z_1"]:
        words = " ".join([f"w{i}" for i in range(counts[name])])
        lines.append(f"{name}: {words}")
    return "\n".join(lines)


def test_valid_content_passes():
    content = _make_content()
    result = verify(content, CONSTRAINTS)
    assert result.passed
    assert result.layers is not None
    assert len(result.layers) == 4


def test_layer_names_parsed():
    content = _make_content()
    result = verify(content, CONSTRAINTS)
    names = [layer.name for layer in result.layers]
    assert names == ["z_4", "z_3", "z_2", "z_1"]


def test_word_counts_parsed():
    content = _make_content()
    result = verify(content, CONSTRAINTS)
    for layer in result.layers:
        assert layer.word_count == CONSTRAINTS[layer.name]


def test_wrong_word_count_fails():
    content = _make_content(z_4=3)  # z_4 should be 2
    result = verify(content, CONSTRAINTS)
    assert not result.passed
    assert any("z_4" in r for r in result.failure_reasons)


def test_missing_layer_fails():
    content = "z_4: hello world\nz_3: a b c d"
    result = verify(content, CONSTRAINTS)
    assert not result.passed


def test_wrong_order_fails():
    lines = [
        "z_3: a b c d",
        "z_4: hello world",
        "z_2: " + " ".join(["w"] * 8),
        "z_1: " + " ".join(["w"] * 16),
    ]
    result = verify("\n".join(lines), CONSTRAINTS)
    assert not result.passed


def test_empty_content_fails():
    result = verify("", CONSTRAINTS)
    assert not result.passed


def test_non_strict_mode():
    content = _make_content(z_4=3)  # wrong count
    result = verify(content, CONSTRAINTS, strict_word_count=False)
    assert result.passed  # non-strict allows mismatch


def test_bad_format_fails():
    result = verify("this is not z_N format at all", CONSTRAINTS)
    assert not result.passed


def test_custom_id_preserved():
    content = _make_content()
    result = verify(content, CONSTRAINTS, custom_id="test-123")
    assert result.custom_id == "test-123"
