"""Joint phase — REINFORCE on q_φ + supervised MLE on p_θ simultaneously."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.generation.dataset import build_rl_parquet
from src.rl.verl_config import build_verl_joint_overrides

log = get_logger(__name__)


def run_joint(
    config: dict[str, Any],
    project_root: Path,
    *,
    config_path: str | Path | None = None,
    wandb_enabled: bool = False,
    extra_overrides: list[str] | None = None,
) -> int:
    """Joint ELBO training: REINFORCE on q_φ + MLE on p_θ in the same loop.

    p_θ is trained inside the ``JointC2FRewardManager``: each batch the reward
    manager computes ``reward = -CE_loss / num_tokens`` for veRL's REINFORCE
    update on q_φ, then does a backward pass and optimiser step on p_θ.

    Expected outcome (posterior collapse sanity check): q_φ collapses to
    degenerate latents while p_θ learns to ignore z and becomes a plain LM
    over x.

    Returns:
        Process returncode (0 on success, 1 if checkpoints are missing).
    """
    from src.rl.train import validate_checkpoint_paths

    rl_joint_cfg = config.get("rl", {}).get("joint", {})
    if not rl_joint_cfg:
        log.error("config['rl']['joint'] section is missing.")
        return 1

    rl_joint_cfg.setdefault("c2f_model_path", "checkpoints/decoder")
    rl_joint_cfg.setdefault("model_path", "checkpoints/sft")

    ok, _ = validate_checkpoint_paths(
        rl_joint_cfg,
        project_root,
        requires=["c2f_model_path", "model_path"],
        next_step_hint={
            "c2f_model_path": "Run step 6 (06_train_decoder.py) first.",
            "model_path": "Run step 4 (04_sft_train.py) first.",
        },
    )
    if not ok:
        return 1

    log.info("Joint: Preparing RL datasets (train + val)...")
    build_rl_parquet(config, project_root, rl_section="joint", split="rl")
    build_rl_parquet(config, project_root, rl_section="joint", split="val")

    overrides = build_verl_joint_overrides(rl_joint_cfg, project_root, wandb_enabled=wandb_enabled)
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    log.info("Joint — REINFORCE on q_φ + MLE on p_θ. Command: %s", " ".join(cmd))

    # Materialise the (CLI-overridden) config to a temp YAML so the reward
    # worker, which reloads it from disk via C2F_CONFIG_PATH, sees the same
    # values as the main process. Pointing at the original YAML drops any
    # ``rl.joint.*`` override on the floor.
    from src.rl.train import materialize_config_for_workers

    worker_config_path = materialize_config_for_workers(config, project_root)
    try:
        env = {
            **os.environ,
            "C2F_CONFIG_PATH": str(worker_config_path),
            # JointC2FRewardManager trains p_θ on GPU inside a Ray worker.
            # Prevent Ray from clearing CUDA_VISIBLE_DEVICES for 0-GPU processes.
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        }
        return subprocess.run(cmd, cwd=project_root, env=env).returncode
    finally:
        worker_config_path.unlink(missing_ok=True)
