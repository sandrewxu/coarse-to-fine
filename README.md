# Coarse-to-Fine Latent Language Model

A pipeline for training a multi-scale language model that generates text through progressively detailed latent representations: z_4 (2 words) -> z_3 (4) -> z_2 (8) -> z_1 (16) -> text (32 words).

## Project Structure

```
coarse-to-fine/
├── config/
│   ├── default.yaml
│   ├── prompts/
│   │   ├── generation.yaml
│   │   └── verification.yaml
│   └── experiments/
│       ├── qwen3_4b_math.yaml
│       └── latent_generation.yaml     # Main experiment config (all steps)
│
├── src/
│   ├── data/
│   │   ├── schemas.py                 # Pydantic models for all pipeline stages
│   │   ├── sharding.py               # Dataset upload, preprocessing
│   │   └── utils.py
│   │
│   ├── batch/                         # Step 2: Batch API client
│   │   ├── client.py
│   │   ├── providers/openai.py
│   │   ├── prompt_builder.py
│   │   └── polling.py
│   │
│   ├── verification/                  # Step 3: Output verification
│   │   ├── verifier.py               # Base verifier interface
│   │   ├── rule_based.py             # Word count + layer structure checks
│   │   ├── model_based.py
│   │   └── pipeline.py
│   │
│   ├── sft/                           # Step 4: Supervised Fine-Tuning
│   │   ├── dataset.py                # Verified outputs -> veRL parquet
│   │   ├── train.py                  # veRL Hydra overrides builder
│   │   └── merge.py                  # LoRA merge utilities
│   │
│   ├── generation/                    # Step 5: Local latent generation
│   │   ├── inference.py              # vLLM offline inference wrapper
│   │   └── dataset.py               # Prompt loading, saving, flattening
│   │
│   ├── qwen3_joint/                   # C2F model architecture
│   │   ├── configuration.py          # C2FConfig (extends Qwen3Config)
│   │   └── modeling.py               # C2FForCausalLM with block-prefix mask
│   │
│   ├── c2f_training/                  # Step 6: C2F pretraining
│   │   ├── dataset.py                # Word-split tokenization pipeline
│   │   └── train.py                  # Model loading + HF Trainer setup
│   │
│   ├── rl/                            # Step 7: RL fine-tuning
│   │   ├── reward.py
│   │   ├── train.py
│   │   └── verl_config.py
│   │
│   └── utils/
│       ├── env.py                   # .env loader + W&B setup
│       ├── logging.py
│       ├── io.py
│       └── cost.py
│
├── scripts/
│   ├── 03_verify_outputs.py           # Verify + create SFT parquet
│   ├── 04_sft_train.py               # veRL SFT via torchrun
│   ├── 05_generate_local.py          # vLLM generation + verification + flatten
│   └── 06_train_decoder.py           # C2F pretraining via HF Trainer
│
├── data/
│   ├── raw/                           # Original datasets
│   ├── prompt_data/                   # Batch API request files (sft.jsonl)
│   ├── shards/
│   ├── batch_outputs/                 # Raw API responses
│   ├── verified/                      # Verification stats
│   ├── sft_dataset/                   # SFT parquet (prompt + response)
│   ├── local_generations/             # Step 5 output (generations + c2f_train)
│   └── decoder_dataset/
│
├── checkpoints/
│   ├── sft/                           # SFT model checkpoints
│   ├── decoder/                       # C2F model checkpoints
│   └── rl/
│
├── .env.example                     # Template for secrets (W&B, OpenAI, HF)
├── .gitignore
├── pyproject.toml
└── Makefile
```

## Pipeline Steps

### Step 3: Verify Batch Outputs

Verifies batch API outputs against latent layer word count constraints and creates the SFT training parquet.

```bash
python scripts/03_verify_outputs.py \
  --input data/batch_outputs/.../output.jsonl \
  --config config/experiments/latent_generation.yaml \
  --prompts data/prompt_data/.../sft.jsonl \
  --output data/verified/latent_generation_10k_v1
```

The `--prompts` flag points to the sft.jsonl file (OpenAI Batch API request format) to extract original 32-word documents as prompts. The output SFT parquet (`data/sft_dataset/train.parquet`) has columns:
- `prompt`: original 32-word document
- `response`: raw `z_4: ...\nz_3: ...\nz_2: ...\nz_1: ...` format

### Step 4: Supervised Fine-Tuning

Runs SFT via veRL on the verified parquet.

```bash
# Install SFT dependencies
pip install -e ".[sft]"

# Run SFT (config controls model, num_gpus 1-4, hyperparams)
python scripts/04_sft_train.py --config config/experiments/latent_generation.yaml
```

### Step 5: Generate Latents from Local Model

Uses vLLM to generate latent outputs from the SFT model at scale, verifies them, and flattens for C2F training.

```bash
# Install generation dependencies
pip install -e ".[generation]"

# Generate (config controls model_path, num_gpus, temperature, etc.)
python scripts/05_generate_local.py \
  --config config/experiments/latent_generation.yaml

# Override sample count or output directory:
python scripts/05_generate_local.py \
  --config config/experiments/latent_generation.yaml \
  --num-samples 100000 \
  --output-dir data/local_generations/run2
```

Produces:
- `data/local_generations/generations.parquet`: raw outputs with `z_n:` format
- `data/local_generations/c2f_train.parquet`: flattened word sequence for C2F training

### Step 6: Pretrain C2F Joint Model

Trains the Coarse-to-Fine model on the flattened latent + text sequences.

```bash
# Install C2F training dependencies
pip install -e ".[c2f]"

# Single GPU:
python scripts/06_train_decoder.py \
  --config config/experiments/latent_generation.yaml

# Multi-GPU with FSDP (set c2f_training.fsdp in config):
accelerate launch --num_processes=4 scripts/06_train_decoder.py \
  --config config/experiments/latent_generation.yaml

# Resume from checkpoint:
python scripts/06_train_decoder.py \
  --config config/experiments/latent_generation.yaml \
  --resume-from checkpoints/decoder/checkpoint-500
```

## Configuration

All pipeline parameters are defined in `config/experiments/latent_generation.yaml`:

- **wandb**: Weights & Biases logging (enabled, project, entity, group, tags, mode)
- **scale_lengths**: Token positions per scale `[2, 4, 8, 16, 32]`
- **word_count_constraints**: Exact word counts per latent layer for verification
- **sft**: Model, num_gpus (1-4), dataset paths, training hyperparams
- **generation**: Model path, vLLM tensor parallelism, sampling params
- **c2f_training**: Init source, FSDP config, training hyperparams

## C2F Model Architecture

The C2F model (`src/qwen3_joint/`) extends Qwen3 with:

- **Block-prefix causal mask**: Token at scale k attends to all tokens at scales < k (not within same scale). BOS is universal.
- **Per-scale position embeddings**: Learned absolute embeddings per scale, replacing RoPE.
- **Unshifted loss**: Each position predicts its own token (not next token), since within-scale tokens are independent.

Sequence layout: `[BOS | z_4 tokens (2) | z_3 tokens (4) | z_2 tokens (8) | z_1 tokens (16) | text tokens (32) | padding]` (total padded to 64).

## Data Flow

```
Batch API outputs (z_n: format)
  -> Step 3: Verify + pair with prompts -> SFT parquet (prompt + response)
  -> Step 4: SFT with veRL -> checkpoints/sft/
  -> Step 5: vLLM generation -> verify -> flatten -> c2f_train.parquet
  -> Step 6: C2F pretraining -> checkpoints/decoder/
  -> Step 7: RL fine-tuning (TODO)
```

## Secrets & Weights and Biases

### Setup

1. Copy the template and fill in your keys:

```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:

```env
# Required for W&B experiment tracking
WANDB_API_KEY=your-wandb-api-key-here

# Optional: override W&B entity (defaults to your W&B account)
WANDB_ENTITY=your-team-or-username

# Required for batch API calls (steps 1-2)
OPENAI_API_KEY=sk-...

# Required for downloading gated models (e.g. Qwen3)
HF_TOKEN=hf_...
```

The `.env` file is gitignored and never committed. On HPC, you can instead export these variables in your job script or module system — existing env vars take precedence over `.env` values.

### W&B Configuration

W&B is controlled by the top-level `wandb:` section in the experiment YAML:

```yaml
wandb:
  enabled: true           # flip to false to disable all W&B logging
  project: "coarse-to-fine"
  entity: null            # uses WANDB_ENTITY from .env or W&B default
  group: "latent-generation"
  tags: ["qwen3-4b"]
  mode: "online"          # "online", "offline", or "disabled"
```

When `wandb.enabled: true`:
- **Step 4 (SFT)**: veRL's trainer logs to W&B via `trainer.logger=['wandb']`
- **Step 5 (Generation)**: Logs verification pass rate, total generated/passed counts
- **Step 6 (C2F)**: HF Trainer's `report_to` is set to `"wandb"` automatically

Each step adds its own tag (e.g. `sft`, `generation`, `c2f-pretrain`) so runs are filterable in the W&B dashboard.

All scripts call `load_env()` + `setup_wandb()` from `src/utils/env.py` early in execution, before any training framework imports.

## Environment

- **Local development**: macOS/Linux, CPU-only testing with small configs
- **HPC training**: 1-4 A100/H100/H200 GPUs
  - Step 4 (SFT): `torchrun --nproc_per_node=N` via veRL
  - Step 5 (Generation): vLLM tensor parallelism (`num_gpus` in config)
  - Step 6 (C2F): `accelerate launch --num_processes=N` with FSDP

Package management via `uv`. Route all changes through the package (`pip install -e .`).
