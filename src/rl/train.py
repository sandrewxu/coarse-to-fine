"""RL phase orchestration entry points (step 7).

This module is a thin dispatcher that re-exports the three phase functions
plus the CLI override parser. Heavy phase logic lives in the sibling
``phase_*.py`` modules:

  - :func:`run_sft_rl`        — Phase A: GRPO on q_φ, p_θ frozen
  - :func:`run_c2f_finetune`  — Phase B: supervised on p_θ, q_φ frozen
  - :func:`run_joint`         — joint posterior + decoder training

``apply_overrides`` lives here (rather than a phase module) because all
three phases route their CLI ``key=value`` overrides through the same parser.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.common.logging import get_logger
from src.rl.phase_c2f_finetune import run_c2f_finetune
from src.rl.phase_joint import run_joint
from src.rl.phase_sft_rl import run_sft_rl

log = get_logger(__name__)

__all__ = [
    "apply_overrides",
    "materialize_config_for_workers",
    "run_c2f_finetune",
    "run_joint",
    "run_sft_rl",
    "validate_checkpoint_paths",
]


def materialize_config_for_workers(config: dict[str, Any], project_root: Path) -> Path:
    """Serialise the (CLI-overridden) config dict to a temp YAML and return its path.

    The RL reward workers live in separate Ray actors and re-read the
    experiment YAML via ``load_exp_config()`` (see ``src/rl/common.py``), which
    resolves the path from the ``C2F_CONFIG_PATH`` env var. If that path
    points at the original on-disk YAML, any ``rl.*`` override applied by
    :func:`apply_overrides` in the main process never reaches the worker —
    the worker sees the file's default values only.

    Round-tripping the already-validated dict through ``yaml.safe_dump`` +
    ``yaml.safe_load`` (done inside ``load_config`` on the worker side) is
    safe because ``load_config`` returns ``model_dump()`` output, which is
    primitive-typed. The caller is responsible for deleting the returned
    temp file after the subprocess finishes.
    """
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="c2f_exp_", dir=str(project_root))
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    return Path(path)


# ── CLI override parsing ─────────────────────────────────────────────────────


def _cast_value(raw: str):
    """Cast a string override value to int / float / bool / None / str."""
    if raw.lower() == "null":
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_overrides(config: dict, overrides: list[str]) -> tuple[dict, list[str]]:
    """Split CLI overrides into config-dict updates (``rl.*``) and veRL pass-throughs.

    ``rl.sft_rl.epochs=1`` updates ``config['rl']['sft_rl']['epochs'] = 1``.
    Everything else is collected as veRL Hydra overrides.

    Returns:
        (updated_config, verl_overrides)
    """
    verl_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            verl_overrides.append(override)
            continue
        key_path, _, raw_value = override.partition("=")
        parts = key_path.split(".")
        if parts[0] != "rl":
            verl_overrides.append(override)
            continue
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _cast_value(raw_value)
    return config, verl_overrides


# ── Shared phase helper ──────────────────────────────────────────────────────


def validate_checkpoint_paths(
    rl_section: dict,
    project_root: Path,
    *,
    requires: list[str],
    next_step_hint: dict[str, str] | None = None,
) -> tuple[bool, dict[str, Path]]:
    """Resolve and validate that required checkpoint paths exist.

    Args:
        rl_section: A subsection of ``config['rl']`` (``sft_rl``, ``joint``, etc).
        project_root: Repo root, used to resolve relative checkpoint paths.
        requires: Keys in ``rl_section`` whose path must exist (e.g. ``["c2f_model_path", "model_path"]``).
        next_step_hint: Optional mapping ``{key → human_hint}`` for "run step N first" messages.

    Returns:
        ``(ok, paths)`` where ``ok`` is True iff every required path exists,
        and ``paths`` maps each key to its resolved absolute Path. When
        ``ok`` is False, ``paths`` is empty and an error has been logged.
    """
    next_step_hint = next_step_hint or {}
    resolved: dict[str, Path] = {}
    for key in requires:
        raw = rl_section.get(key)
        if raw is None:
            log.error("config is missing %r", key)
            return False, {}
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            log.error("%s not found: %s", key, path)
            if hint := next_step_hint.get(key):
                log.error(hint)
            return False, {}
        resolved[key] = path
    return True, resolved
