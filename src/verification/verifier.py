"""
Space-based verification for latent layer outputs.

Verifies that generated text has the expected z_N: layer format with
correct word counts (splitting on whitespace).
"""
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Layer:
    """A parsed latent layer."""
    name: str
    content: str
    word_count: int


@dataclass
class VerificationResult:
    """Result of verifying a single output."""
    custom_id: str
    passed: bool = True
    raw_content: str = ""
    layers: list[Layer] | None = None
    failure_reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str):
        self.failure_reasons.append(reason)
        self.passed = False


# Expected layer order (coarse to fine)
_EXPECTED_LAYERS = ["z_4", "z_3", "z_2", "z_1"]
_LAYER_RE = re.compile(r"^(z_\d+):\s*(.*)$")


def verify(
    content: str,
    word_count_constraints: dict[str, int],
    custom_id: str = "",
    strict_word_count: bool = True,
) -> VerificationResult:
    """
    Verify that content has valid z_N: layer format with correct word counts.

    Uses space-based word splitting. Each non-empty line must match
    ``z_N: <words>``, layers must appear in [z_4, z_3, z_2, z_1] order,
    and word counts must match constraints.

    Args:
        content: Raw text to verify (e.g. "z_4: hello world\\nz_3: ...")
        word_count_constraints: e.g. {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}
        custom_id: Identifier for tracking
        strict_word_count: If True, word counts must match exactly

    Returns:
        VerificationResult with pass/fail and parsed layers
    """
    result = VerificationResult(custom_id=custom_id, raw_content=content)

    # Parse layers
    layers = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _LAYER_RE.match(line)
        if not match:
            result.fail("Failed to parse layer structure")
            return result
        name = match.group(1)
        layer_content = match.group(2).strip()
        words = layer_content.split()
        layers.append(Layer(name=name, content=layer_content, word_count=len(words)))

    if not layers:
        result.fail("No layers found")
        return result

    # Check layer count and order
    layer_names = [l.name for l in layers]
    if layer_names != _EXPECTED_LAYERS:
        result.fail(
            f"Expected layers {_EXPECTED_LAYERS}, got {layer_names}"
        )
        return result

    # Check word counts
    for layer in layers:
        expected = word_count_constraints.get(layer.name)
        if expected is None:
            result.fail(f"No word count constraint for {layer.name}")
            return result
        if strict_word_count and layer.word_count != expected:
            result.fail(
                f"{layer.name}: expected {expected} words, got {layer.word_count}"
            )
            return result

    # Check for empty layers
    for layer in layers:
        if not layer.content:
            result.fail(f"{layer.name} has empty content")
            return result

    result.layers = layers
    return result
