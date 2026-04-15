"""
Wrapper for verl.trainer.main_ppo that forces gloo backend.

On HPC nodes where the CUDA driver is too old for the NCCL version bundled
with PyTorch, FSDP init crashes during _sync_module_params_and_buffers
(which calls dist._broadcast_coalesced via NCCL).  For single-GPU runs
this broadcast is a no-op, so gloo works fine as a substitute.

Usage (replaces ``python -m verl.trainer.main_ppo``):
    python src/rl/joint_launcher.py <hydra overrides...>
"""
import runpy
import sys

import torch.distributed as dist

_orig_init_process_group = dist.init_process_group


def _gloo_init_process_group(*args, **kwargs):
    """Replace nccl backend with gloo."""
    if args:
        args = ("gloo",) + args[1:]
    else:
        kwargs["backend"] = "gloo"
    return _orig_init_process_group(*args, **kwargs)


dist.init_process_group = _gloo_init_process_group

# Run veRL's main_ppo exactly as ``python -m verl.trainer.main_ppo`` would.
# alter_sys=True updates sys.argv[0] so Hydra sees the correct module path.
runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)
