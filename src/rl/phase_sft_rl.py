"""Phase A — GRPO on q_φ (SFT model) with p_θ (C2F) frozen."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.generation.dataset import build_rl_parquet
from src.rl.verl_config import build_verl_grpo_overrides

log = get_logger(__name__)


def run_sft_rl(
    config: dict[str, Any],
    project_root: Path,
    *,
    config_path: str | Path | None = None,
    wandb_enabled: bool = False,
    extra_overrides: list[str] | None = None,
) -> int:
    """Phase A: GRPO on q_φ with p_θ frozen.

    Steps:
      1. Validate that required checkpoints exist.
      2. Prepare the RL parquet (add ground_truth / data_source columns).
      3. Build GRPO Hydra overrides via :func:`build_verl_grpo_overrides`.
      4. Launch ``python -m verl.trainer.main_ppo <overrides>``.

    The ``C2F_CONFIG_PATH`` environment variable is set before launching so
    ``C2FRewardManager.__init__`` can locate the experiment YAML inside the
    veRL worker (which does not inherit the parent CWD reliably).

    Returns:
        Process returncode (0 on success, 1 if checkpoints are missing).
    """
    from src.rl.train import validate_checkpoint_paths

    rl_sft_cfg = config.get("rl", {}).get("sft_rl", {})
    if not rl_sft_cfg:
        log.error("config['rl']['sft_rl'] section is missing.")
        return 1

    rl_sft_cfg.setdefault("c2f_model_path", "checkpoints/decoder")
    rl_sft_cfg.setdefault("model_path", "checkpoints/sft")

    ok, _ = validate_checkpoint_paths(
        rl_sft_cfg,
        project_root,
        requires=["c2f_model_path", "model_path"],
        next_step_hint={
            "c2f_model_path": "Run step 6 (06_train_decoder.py) first to pretrain the C2F model.",
            "model_path": "Run step 4 (04_sft_train.py) first.",
        },
    )
    if not ok:
        return 1

    log.info("Phase A: Preparing RL dataset...")
    build_rl_parquet(config, project_root, rl_section="sft_rl")

    overrides = build_verl_grpo_overrides(rl_sft_cfg, project_root, wandb_enabled=wandb_enabled)
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    log.info("Phase A — GRPO on q_φ. Command: %s", " ".join(cmd))

    # Serialise the (CLI-overridden) config so the reward worker sees the
    # same values as the main process — the worker reloads from disk via
    # C2F_CONFIG_PATH and won't pick up any in-memory ``rl.sft_rl.*`` override.
    from src.rl.train import materialize_config_for_workers

    worker_config_path = materialize_config_for_workers(config, project_root)
    try:
        env = {**os.environ, "C2F_CONFIG_PATH": str(worker_config_path)}
        return subprocess.run(cmd, cwd=project_root, env=env).returncode
    finally:
        worker_config_path.unlink(missing_ok=True)
