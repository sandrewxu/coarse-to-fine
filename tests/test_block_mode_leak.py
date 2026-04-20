"""Correctness tests for block-mode C2F forward.

The block decoder uses unshifted CE (``logits[i]`` predicts ``labels[i] =
input_ids[i]``), which would make the residual stream a trivial leak channel
from ``input_ids[i]`` to ``logits[i]``. ``C2FModel.forward`` kills the leak
by overriding the layer-0 post-attention residual at content positions with
``mask_embedding + scale_pos_emb[i]``, while keeping real embeddings on the
Q/K/V path so earlier-scale attention actually carries per-example info.

The three invariants under test:

1. **No self-leak.** Toggling ``input_ids[i]`` at one content position must
   leave ``logits[:, i]`` bitwise-close — else the residual-stream leak
   still exists and unshifted CE is trivially satisfiable.
2. **Cross-scale flow.** Toggling ``input_ids[j]`` at a scale-0 position
   must change ``logits[:, i]`` at later scales — else the mask_embedding
   fix has eaten the K/V content (the bug this fix targets).
3. **Same-scale independence.** Toggling ``input_ids[j]`` at one scale-1
   position must not change ``logits[:, i]`` at another scale-1 position —
   the block mask forbids same-scale attention.

Causal-mode regression: a sanity check that ``mask_type="causal"`` still
produces valid outputs and routes through ``create_causal_mask`` (not
``create_c2f_block_causal_mask``).
"""

import pytest
import torch

pytest.importorskip("transformers")

from src.c2f_model.configuration import C2FConfig
from src.c2f_model.modeling import C2FForCausalLM

SCALE_LENGTHS = [2, 3, 4]  # content positions 1..9 (BOS at 0)
SEQ_LEN = 1 + sum(SCALE_LENGTHS)  # 10; no padding so every position is content
VOCAB = 100


def _tiny_c2f(mask_type: str) -> C2FForCausalLM:
    cfg = C2FConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        max_position_embeddings=SEQ_LEN,
        scale_lengths=SCALE_LENGTHS,
        mask_type=mask_type,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return C2FForCausalLM(cfg).eval()


def _scale_of(pos: int) -> int:
    """Scale index for a given 0-based position; BOS (0) has scale -1."""
    if pos == 0:
        return -1
    running = 1
    for k, length in enumerate(SCALE_LENGTHS):
        if running <= pos < running + length:
            return k
        running += length
    raise ValueError(f"pos {pos} out of content range")


def _baseline_ids() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(5, VOCAB - 5, (1, SEQ_LEN), dtype=torch.long)


def _logits(model: C2FForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids).logits  # (1, T, V)


def test_block_mode_no_self_leak():
    """Toggling input_ids[i] must leave logits[:, i] unchanged in block mode."""
    model = _tiny_c2f("block")
    ids_a = _baseline_ids()

    # Test one content position in each scale.
    for pos in (1, 4, 7):  # scale 0, scale 1, scale 2
        ids_b = ids_a.clone()
        ids_b[0, pos] = (int(ids_a[0, pos].item()) + 17) % VOCAB
        logits_a = _logits(model, ids_a)
        logits_b = _logits(model, ids_b)
        diff = (logits_a[0, pos] - logits_b[0, pos]).abs().max().item()
        assert diff < 1e-5, (
            f"Leak at pos={pos} (scale {_scale_of(pos)}): max |Δlogits[pos]|={diff:.2e}"
        )


def test_block_mode_cross_scale_flow():
    """Toggling an earlier-scale input_id must change a later-scale logit."""
    model = _tiny_c2f("block")
    ids_a = _baseline_ids()
    ids_b = ids_a.clone()
    # Toggle a scale-0 token (position 1 or 2).
    ids_b[0, 1] = (int(ids_a[0, 1].item()) + 23) % VOCAB

    logits_a = _logits(model, ids_a)
    logits_b = _logits(model, ids_b)

    # Later-scale positions must see the change.
    for later_pos in (3, 4, 7, 8):  # scale 1 and scale 2 positions
        diff = (logits_a[0, later_pos] - logits_b[0, later_pos]).abs().max().item()
        assert diff > 1e-3, (
            f"Cross-scale flow missing: toggling pos=1 (scale 0) did not "
            f"move logits at pos={later_pos} (scale {_scale_of(later_pos)}); "
            f"max |Δ|={diff:.2e}. K/V content is being wiped."
        )


def test_block_mode_same_scale_independence():
    """Toggling a scale-k input_id must NOT change another scale-k position's logits."""
    model = _tiny_c2f("block")
    ids_a = _baseline_ids()
    ids_b = ids_a.clone()
    # Toggle position 3 (scale 1). Positions 4, 5 are also scale 1.
    ids_b[0, 3] = (int(ids_a[0, 3].item()) + 11) % VOCAB

    logits_a = _logits(model, ids_a)
    logits_b = _logits(model, ids_b)

    for same_scale_pos in (4, 5):
        diff = (logits_a[0, same_scale_pos] - logits_b[0, same_scale_pos]).abs().max().item()
        assert diff < 1e-5, (
            f"Same-scale leak: toggling pos=3 (scale 1) moved logits at "
            f"pos={same_scale_pos} (scale 1); max |Δ|={diff:.2e}"
        )


def test_block_mode_earlier_scale_unaffected():
    """Toggling a later-scale token must NOT change earlier-scale logits (block mask is strict)."""
    model = _tiny_c2f("block")
    ids_a = _baseline_ids()
    ids_b = ids_a.clone()
    # Toggle position 7 (scale 2). Position 3 (scale 1) and 1 (scale 0) must not move.
    ids_b[0, 7] = (int(ids_a[0, 7].item()) + 13) % VOCAB

    logits_a = _logits(model, ids_a)
    logits_b = _logits(model, ids_b)

    for earlier_pos in (1, 2, 3):
        diff = (logits_a[0, earlier_pos] - logits_b[0, earlier_pos]).abs().max().item()
        assert diff < 1e-5, (
            f"Earlier scale contaminated: toggling pos=7 (scale 2) moved "
            f"logits at pos={earlier_pos}; max |Δ|={diff:.2e}"
        )


def test_causal_mode_runs_and_uses_earlier_context():
    """Regression: causal mode forwards successfully and earlier tokens influence later logits."""
    model = _tiny_c2f("causal")
    ids_a = _baseline_ids()
    ids_b = ids_a.clone()
    ids_b[0, 2] = (int(ids_a[0, 2].item()) + 19) % VOCAB

    logits_a = _logits(model, ids_a)
    logits_b = _logits(model, ids_b)

    assert logits_a.shape == (1, SEQ_LEN, VOCAB)
    # Earlier positions (<2) unchanged.
    for earlier in (0, 1):
        diff = (logits_a[0, earlier] - logits_b[0, earlier]).abs().max().item()
        assert diff < 1e-5, f"Causal should not leak backward at pos={earlier}"
    # Later positions (>=2) should see the change (position 2's token feeds forward).
    diff = (logits_a[0, 5] - logits_b[0, 5]).abs().max().item()
    assert diff > 1e-3, "Causal mode should propagate earlier token changes forward"
