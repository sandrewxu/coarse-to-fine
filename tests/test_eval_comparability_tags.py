"""Pin the ``comparable_to`` invariant across evaluators.

Three evaluators report nats per text word bounding ``-log p(x)`` and must
carry ``comparable_to="neg_log_p_x_per_text_word"``:

- :func:`src.eval.ar.eval_ar`         (exact)
- :func:`src.eval.diffusion.eval_diffusion`  (MC upper bound)
- :func:`src.eval.bound.eval_c2f_bound`      (ELBO upper bound, K=1)

One evaluator reports the C2F joint training loss per content token and must
carry ``comparable_to="joint_train_loss_per_token"``:

- :func:`src.eval.c2f.eval_c2f`

Downstream tooling (plotters, comparison helpers, ultimately a paper table)
reads these tags to refuse invalid cross-axis comparisons. If the tag moves or
changes value without the intent of moving axes, that's the bug this test
catches.

We inspect the source AST rather than instantiating the evaluators because
loading real checkpoints in a unit test is out of scope.
"""

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent


def _final_result_dict_literal(src_path: Path) -> dict[str, object]:
    """Return the last top-level dict literal in the module source.

    Every evaluator builds its return dict as a literal in the last statement
    of its entry function; we parse the file and return that literal's
    constant-valued entries. Non-constant values (like ``float(...)`` calls)
    are represented as the string ``"<expr>"`` — we only need to assert
    on the literal-typed fields anyway.
    """
    tree = ast.parse(src_path.read_text())
    dicts = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
    # We want the dict that has the ``comparable_to`` key — not the per-doc
    # row dicts (list comprehension builders) or intermediate accumulators.
    for node in reversed(dicts):
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "comparable_to" in keys:
            out: dict[str, object] = {}
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    out[k.value] = v.value
            return out
    raise AssertionError(f"{src_path} has no result dict with a 'comparable_to' key.")


@pytest.mark.parametrize(
    ("rel_path", "expected_model_kind"),
    [
        ("src/eval/ar.py", "ar"),
        ("src/eval/diffusion.py", "diffusion"),
        ("src/eval/bound.py", "c2f_bound"),
    ],
)
def test_marginal_nll_evaluators_share_tag(rel_path, expected_model_kind):
    result = _final_result_dict_literal(_PROJECT_ROOT / rel_path)
    assert result["model_kind"] == expected_model_kind
    assert result["comparable_to"] == "neg_log_p_x_per_text_word", (
        f"{rel_path} must tag its output as comparable on the per-text-word "
        "axis; otherwise the AR/diffusion/c2f-bound three-way comparison is "
        "no longer self-consistent."
    )
    assert result["denominator"] == "text_word"


def test_c2f_train_loss_is_not_comparable_to_marginal_evaluators():
    result = _final_result_dict_literal(_PROJECT_ROOT / "src/eval/c2f.py")
    assert result["model_kind"] == "c2f"
    assert result["comparable_to"] == "joint_train_loss_per_token", (
        "eval_c2f reports the joint training objective (-log p(x, z) per "
        "content token), NOT a bound on -log p(x) per text word. Its "
        "comparable_to tag must stay distinct from the marginal evaluators, "
        "or downstream tooling will unwittingly plot train-loss next to AR."
    )
    assert result["denominator"] == "joint_content_tokens"


def test_no_evaluator_uses_an_unknown_comparable_to_value():
    """All current evaluators must use one of the two sanctioned axes.

    A third axis is a design decision, not a casual rename — if you're adding
    one, update this test deliberately so the reviewer sees it.
    """
    allowed = {"neg_log_p_x_per_text_word", "joint_train_loss_per_token"}
    for rel_path in (
        "src/eval/ar.py",
        "src/eval/diffusion.py",
        "src/eval/bound.py",
        "src/eval/c2f.py",
    ):
        result = _final_result_dict_literal(_PROJECT_ROOT / rel_path)
        assert result["comparable_to"] in allowed, (
            f"{rel_path} introduced a new comparable_to value "
            f"{result['comparable_to']!r}; add it to the allow-list here if "
            "that's intentional."
        )
