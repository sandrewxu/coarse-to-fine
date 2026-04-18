"""Pure-function tests for ``src.rl.common``.

Covers ``parse_layers``, ``compute_word_boundaries``, ``build_c2f_input``,
and ``strip_think``. No CUDA, no veRL — these run in CI on a clean Python
install with just ``torch`` and ``pydantic``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from src.rl.common import (
    build_c2f_input,
    compute_word_boundaries,
    parse_layers,
    strip_think,
)

CONSTRAINTS = {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}
SCALE_LENGTHS = [2, 4, 8, 16, 32]
TEXT_WORDS = 32


def _good_response() -> str:
    return (
        "z_4: alpha beta\n"
        "z_3: gamma delta epsilon zeta\n"
        "z_2: eta theta iota kappa lambda mu nu xi\n"
        "z_1: " + " ".join(f"w{i}" for i in range(16)) + "\n"
    )


def test_strip_think_removes_block():
    assert strip_think("<think>scratch</think>real") == "real"


def test_strip_think_handles_multiline():
    text = "before<think>line1\nline2</think>\nafter"
    assert strip_think(text) == "before\nafter"


def test_strip_think_no_block_returns_stripped():
    assert strip_think("  hello  ") == "hello"


def test_parse_layers_returns_contents_in_z4_to_z1_order():
    layers = parse_layers(_good_response(), CONSTRAINTS, strict=True)
    assert layers is not None
    assert len(layers) == 4
    assert layers[0] == "alpha beta"
    assert layers[1] == "gamma delta epsilon zeta"
    assert layers[2] == "eta theta iota kappa lambda mu nu xi"


def test_parse_layers_returns_none_for_wrong_word_count_in_strict_mode():
    bad = _good_response().replace("alpha beta", "alpha")  # z_4: 1 word
    assert parse_layers(bad, CONSTRAINTS, strict=True) is None


def test_parse_layers_returns_none_for_missing_layer():
    bad = _good_response().replace("z_3:", "z_X:")
    assert parse_layers(bad, CONSTRAINTS, strict=True) is None


def test_parse_layers_strips_think_block():
    text = "<think>scratchpad</think>\n" + _good_response()
    assert parse_layers(text, CONSTRAINTS, strict=True) is not None


def test_compute_word_boundaries_layout():
    boundaries = compute_word_boundaries(CONSTRAINTS, TEXT_WORDS)
    assert boundaries == [(0, 2), (2, 6), (6, 14), (14, 30), (30, 62)]


def test_compute_word_boundaries_total_matches_layout():
    boundaries = compute_word_boundaries(CONSTRAINTS, TEXT_WORDS)
    assert boundaries[-1][1] == sum(CONSTRAINTS.values()) + TEXT_WORDS


class _FakeTokenizer:
    """Minimal stand-in for the space tokenizer in tests.

    Maps each word to a small integer id (1-based to keep 0 free for pad).
    Real tokenizer behaviour: ``add_special_tokens=False`` returns one id per
    space-separated word.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return ids


def _common_input_args() -> dict:
    boundaries = compute_word_boundaries(CONSTRAINTS, TEXT_WORDS)
    return {
        "scale_lengths": SCALE_LENGTHS,
        "word_boundaries": boundaries,
        "space_tokenizer": _FakeTokenizer(),
        "bos_id": 1000,
        "pad_id": 0,
        "seq_len": 64,  # 2 ** ceil(log2(1 + 62)) = 64
    }


def test_build_c2f_input_shapes_and_bos():
    layer_contents = [
        "alpha beta",
        "gamma delta epsilon zeta",
        "eta theta iota kappa lambda mu nu xi",
        " ".join(f"w{i}" for i in range(16)),
    ]
    prompt = " ".join(f"t{i}" for i in range(TEXT_WORDS))
    args = _common_input_args()
    input_ids, labels = build_c2f_input(layer_contents, prompt, label_strategy="sft", **args)

    assert input_ids.shape == (args["seq_len"],)
    assert labels.shape == input_ids.shape
    assert input_ids[0].item() == args["bos_id"]
    # BOS position always masked
    assert labels[0].item() == -100


def test_build_c2f_input_sft_masks_pad_in_labels():
    """In sft mode, every pad_id position in input_ids should be -100 in labels."""
    layer_contents = ["a b", "c d e f", "g h i j k l m n", " ".join("o" * 16)]
    args = _common_input_args()
    input_ids, labels = build_c2f_input(
        layer_contents, "short prompt", label_strategy="sft", **args
    )
    pad_positions = (input_ids == args["pad_id"]).nonzero(as_tuple=True)[0]
    assert (labels[pad_positions] == -100).all()


def test_build_c2f_input_joint_masks_post_content_region():
    """In joint mode, the post-content region (after 1 + sum(scale_lengths)) is masked."""
    layer_contents = ["a b", "c d e f", "g h i j k l m n", " ".join("o" * 16)]
    args = _common_input_args()
    input_ids, labels = build_c2f_input(
        layer_contents, " ".join(f"t{i}" for i in range(TEXT_WORDS)), label_strategy="joint", **args
    )
    content_len = 1 + sum(SCALE_LENGTHS)
    assert (labels[content_len:] == -100).all()
    # In-block padding should NOT be masked under joint
    # (specifically, positions [1:content_len] keep their input id even if pad)
    assert labels[1:content_len].tolist() == input_ids[1:content_len].tolist()


def test_build_c2f_input_unknown_label_strategy_raises():
    with pytest.raises(ValueError):
        build_c2f_input(
            ["a", "b c", "d e f g h", " ".join("x" * 16)],
            "p",
            label_strategy="bogus",  # type: ignore[arg-type]
            **_common_input_args(),
        )


def test_build_c2f_input_pads_short_segments():
    """When a layer has fewer words than scale_lengths[k], pad_id fills the gap."""
    layer_contents = ["", "", "", ""]  # all empty layers
    args = _common_input_args()
    input_ids, _ = build_c2f_input(layer_contents, "", label_strategy="sft", **args)
    # First token is BOS, the rest are pad
    assert input_ids[0].item() == args["bos_id"]
    assert (input_ids[1:] == args["pad_id"]).all()


def test_build_c2f_input_truncates_long_segments():
    """When a layer's encoded length exceeds scale_lengths[k], it's truncated."""
    # Make z_4 have 5 words but scale_lengths[0] = 2 → only 2 tokens land
    layer_contents = ["one two three four five", "x y z w", "a b c d e f g h", " ".join("u" * 16)]
    args = _common_input_args()
    input_ids, _ = build_c2f_input(layer_contents, "p", label_strategy="sft", **args)
    # Position 0 = BOS, positions 1-2 are the two truncated z_4 tokens
    z4_token_count = SCALE_LENGTHS[0]
    # The z_4 segment occupies positions [1:1+z4_token_count]
    z4_segment = input_ids[1 : 1 + z4_token_count]
    assert (z4_segment != args["pad_id"]).all()  # all real tokens, no padding


def test_build_c2f_input_returns_long_tensors():
    args = _common_input_args()
    input_ids, labels = build_c2f_input(
        ["a b", "c d e f", "g h i j k l m n", " ".join("o" * 16)],
        "p",
        label_strategy="sft",
        **args,
    )
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
