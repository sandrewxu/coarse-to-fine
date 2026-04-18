# `src/rl/` — RL / ELBO optimisation (step 7)

Step 7 alternates between updating the posterior $q_\phi$ (the SFT model) and
the joint generative model $p_\theta$ (the C2F decoder), to optimise the ELBO
without collapsing the posterior. The entry script is
[`scripts/07_rl_train.py`](../../scripts/07_rl_train.py); the README's "Step 7"
section has full invocations.

## Phases

| Phase | Updated | Frozen | Reward manager | Trainer |
|---|---|---|---|---|
| **A — `sft`** | $q_\phi$ (SFT) | $p_\theta$ (C2F) | `C2FRewardManager` | veRL GRPO |
| **B — `c2f`** | $p_\theta$ (C2F) | $q_\phi$ (SFT) | n/a (supervised) | HF Trainer |
| **Joint** | both | none | `JointC2FRewardManager` | veRL GRPO + custom optimiser |

`--phase both` runs A then B sequentially. Phase A and Joint depend on veRL;
Phase B is pure HuggingFace Trainer + FSDP.

## File map

| File | Purpose |
|---|---|
| `reward.py` | `C2FRewardManager` (Phase A) and `JointC2FRewardManager` (Joint). Called by veRL during rollout scoring. **High-value refactor target** — the two classes share ~200 lines of helper code that will be lifted into `common.py` in a later phase. |
| `train.py` | Phase orchestration (`run_sft_rl`, `run_c2f_finetune`, `run_joint`) and `apply_overrides` for CLI key=value flags. |
| `verl_config.py` | Builds Hydra-style config overrides from the experiment YAML for invoking the veRL trainer subprocess. |

## The `C2F_CONFIG_PATH` contract

veRL spawns workers via Ray with a stripped environment that does not reliably
inherit the parent process's CWD or env. The reward managers therefore re-locate
the experiment YAML via the `C2F_CONFIG_PATH` env var:

```python
config_path = os.environ.get("C2F_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
```

The launch wrappers (`scripts/slurm_07_rl.sh` and the in-process subprocess in
`run_sft_rl`) export `C2F_CONFIG_PATH` before `torchrun`. If you see a "config
not found" error from a reward manager, check that this env var is set.

## veRL upstream coupling

The reward managers inherit from
`verl.experimental.reward_loop.reward_manager.base.RewardManagerBase`. If veRL
moves or renames that class, the imports in `reward.py:30` will break loudly —
that's the canary. `pyproject.toml` pins `verl>=0.7.0` from PyPI; bump
deliberately and re-run the integration smoke (`scripts/slurm_07_rl.sh` with
`trainer.total_epochs=1`) when upgrading.
