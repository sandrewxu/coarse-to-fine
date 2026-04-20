# Claude / agent context — coarse-to-fine

This repo trains a **coarse-to-fine hierarchical latent LM**: a Qwen3-4B-based VAE
where the latents `z_4 → z_3 → z_2 → z_1 → x` are natural-language summaries of
increasing detail (2, 4, 8, 16, 32 words). The full algorithm, ELBO derivation, and
attention-mask diagrams live in `README.md` — read that first for the *what* and
*why*. This file covers *how to work in the repo*.

## Repo map

- `src/` — pipeline modules. See `src/c2f_model/README.md` (model fork) and
  `src/rl/README.md` (RL phases + veRL coupling) for the two non-obvious folders.
- `scripts/` — numbered `00_…_09_*.py` entry points, one per pipeline step.
  Matching `slurm_*.sh` wrappers for HPC submission.
- `config/latent_generation.yaml` — only non-default values; defaults live in
  `src/config.py` (Pydantic schema).
- `prompts/{system_prompts,user_prompts,few_shot_examples}/` — batch-API templates,
  referenced by name from `config.batch.*`.
- `tests/` — pytest, run with `make test`.
- `notebooks/` — scratch notebooks for ad-hoc analysis. Outputs are stripped on
  commit by `nbstripout` (`.pre-commit-config.yaml`).

## Common commands

```bash
make install     # uv sync --extra dev + pre-commit install
make lint        # ruff check . + ruff format --check .
make format      # ruff format . + ruff check --fix .
make test        # pytest tests/

# Per-step optional groups (GPU only):
pip install -e ".[sft]"        # step 4 — HF Trainer (accelerate + torch)
pip install -e ".[generation]" # step 5 — vLLM
pip install -e ".[c2f]"        # step 6 — HF Trainer + FSDP (+ liger-kernel)
pip install -e ".[rl]"         # step 7 — veRL (+ flash-attn; needs --no-build-isolation)

# Run a pipeline step:
python scripts/0N_<name>.py --config config/latent_generation.yaml
# SLURM equivalent:
sbatch scripts/slurm_0N_*.sh
```

## Key conventions (load-bearing — preserve these when editing)

- **`src/config.py` is the single source of truth for defaults.** YAML overrides
  only what differs; CLI flags (script 07 only) override YAML. Do **not** add
  YAML-only fields without adding a Pydantic field — the loader silently drops them.
- **`# C2F:` annotations.** Every line in `src/c2f_model/modeling.py` that
  diverges from upstream `transformers.models.qwen3.modeling_qwen3` is marked
  `# C2F:`. Preserve this when editing — it's how the upstream diff is auditable.
  When you *remove* a line, leave a `# C2F: removed — <original>` stub.
- **`load_env()` then `setup_wandb(config)` BEFORE `import torch / transformers / verl`.**
  Both `src/common/env.py` helpers must run before the heavy imports — see any
  script in `scripts/` for the pattern.
- **Space tokenizer = one word per token.** This is what guarantees scale
  boundaries land at deterministic positions. Do not replace it without
  rewriting `C2FDataset` and the C2F input builders in `src/rl/common.py`.

## Gotchas (things that have bitten us)

- **Flash Attention 2 is intentionally disabled** (`attn_implementation="eager"`).
  FA2 silently ignores additive masks, which would break block-prefix attention.
- **Manual coupling between scripts.** After running script 01, you must update
  `sft.prompt_data` in the YAML to point at the new output path before running
  script 03.
- **veRL workers don't inherit the parent env reliably.** The experiment YAML is
  re-located inside the worker via the `C2F_CONFIG_PATH` env var (set by the
  SLURM/launch script before `torchrun`). If you see "config not found" errors in
  reward managers, check that env var. The lookup helper is `src.rl.common.load_exp_config`.
- **Two attention modes coexist:** `mask_type="block"` (default, C2F semantics)
  and `mask_type="causal"` (AR baseline). Step 8 and 9 evaluators handle both —
  if you change the model, both code paths must keep working.
- **transformers 4.51 vs 4.52 compatibility.** `modeling.py` has try/except shims
  for `auto_docstring`, `capture_outputs`, and `merge_with_config_defaults`. Do
  not remove the fallbacks — both versions are in use.
- **`verl` upstream is moving.** `pyproject.toml` pins `verl>=0.7.0` (PyPI). If
  veRL refactors `RewardManagerBase` (currently at
  `verl.experimental.reward_loop.reward_manager.base`), the imports in
  `src/rl/reward_sft.py` and `src/rl/reward_joint.py` are the canary — they'll
  break loudly.

## Where to add things

| Goal | Where |
|---|---|
| New pipeline step | numbered `scripts/NN_*.py` + matching `slurm_NN_*.sh` + new section in `README.md` "Running the Pipeline" + new Pydantic section in `src/config.py` |
| New dataset | register in `src/data/registry.py` + add a preprocess fn in `src/data/preprocessing.py` |
| New RL reward shaping | edit `src/rl/reward_sft.py` (Phase A) or `src/rl/reward_joint.py` (joint); shared helpers live in `src/rl/common.py`. Phase dispatch is in `scripts/07_rl_train.py` (`--phase {sft,c2f,joint,both}`). |
| New attention variant | edit `src/c2f_model/modeling.py`, add a `create_*_mask` function, gate via `C2FConfig.mask_type`, mark every change with `# C2F:` |
| New batch-API prompt | drop a file under `prompts/{system_prompts,user_prompts,few_shot_examples}/` and reference it by filename in `config.batch` |

## Do not

- Commit `data/`, `checkpoints/`, `.env`, `wandb/` — all gitignored.
- Add YAML-only config fields (loader silently drops them).
- Reintroduce RoPE in `C2FAttention` — positions come from `C2FScaleEmbedding`.
- Run `--no-verify` to bypass pre-commit. If a hook fails, fix the underlying issue.
- Skip the `# C2F:` annotation when modifying `src/c2f_model/modeling.py`.
