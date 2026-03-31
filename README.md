# Coarse-to-Fine Latent Language Model

Most discrete diffusion approaches use a **fixed** forward process — independently masking tokens, adding Gaussian noise in embedding space, or applying some other corruption schedule that has no semantic meaning. The reverse process must then "undo" corruption steps that were never designed to be meaningful.

This project takes a different route: **learn** the forward and reverse processes jointly by optimising the ELBO, as in a variational autoencoder, while making use of the strong natural-language prior of a pretrained autoregressive LM. The result is a coarse-to-fine generative process that first produces a high-level document summary $z_T$, then a slightly more detailed one $z_{T-1}$, and so on until generating a coherent document $x$. Because the hierarchy is learned rather than fixed, each latent level has a well-defined semantic interpretation.

---

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [Algorithm](#algorithm)
3. [Theoretical Background](#theoretical-background)
4. [Model Architecture](#model-architecture)
5. [Pipeline Overview](#pipeline-overview)
6. [Setup](#setup)
7. [Running the Pipeline](#running-the-pipeline)
8. [Configuration](#configuration)
9. [Project Layout](#project-layout)
10. [Monitoring with W&B](#monitoring-with-wb)

---

## Background and Motivation

### Why not standard diffusion?

Continuous diffusion (DDPM-style) adds Gaussian noise; masked diffusion independently flips tokens to a `[MASK]` state. Both forward processes are **fixed and semantically arbitrary** — the reverse process must reconstruct text from noise that was never designed to be meaningful at intermediate steps.

### This approach: learned coarse-to-fine hierarchy

We instead use a prompted autoregressive LM (e.g. a strong reasoning model) to generate a coarse-to-fine latent hierarchy for each training document. The hierarchy is a sequence of increasingly detailed textual summaries:

$$z_T \;\to\; z_{T-1} \;\to\; \cdots \;\to\; z_1 \;\to\; x$$

where $z_T$ is the coarsest (fewest words) and $x$ is the original document. Each latent level has a natural semantic interpretation as a "summary at resolution $k$."

A VAE-like training objective (the ELBO) then jointly tightens the approximate posterior $q_\phi(z \mid x)$ and the joint generative model $p_\theta(z, x)$.

### Key design choices in this repo

- **Block-prefix causal attention**: the joint model $p_\theta$ uses a transformer with a custom attention mask so that tokens at scale $k$ attend to all coarser scales but are independent within the same scale. This is in the spirit of [VAR](https://arxiv.org/abs/2404.02905) and [Block Diffusion](https://arxiv.org/abs/2503.09573).
- **Alternating ELBO optimisation**: $q_\phi$ is updated with GRPO (REINFORCE) while $p_\theta$ is frozen, then $p_\theta$ is updated with a direct supervised gradient while $q_\phi$ is frozen. This prevents posterior collapse ([Bowman et al., 2015](https://arxiv.org/abs/1511.06349)).
- **Space tokenizer**: a word-level tokenizer (one word = one token) guarantees that scale boundaries in the sequence are always at deterministic positions, making the fixed-layout C2F model viable.

### Background reading

- [DDPM — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [D3PM — Structured Denoising Diffusion Models](https://arxiv.org/abs/2107.03006)
- [VAR — Visual AutoRegressive Modeling](https://arxiv.org/abs/2404.02905)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
- [Non-Markovian Discrete Diffusion with Causal Language Models](https://arxiv.org/abs/2502.09767)
- [Preventing Posterior Collapse with delta-VAEs](https://arxiv.org/abs/1901.05534)
- GRPO: [DeepSeekMath](https://arxiv.org/abs/1906.02691); CS336 Assignment 5

---

## Algorithm

### Phase 0 — Initialise $q_\phi$ and $p_\theta$

**Step 1 — Synthetic latent generation (upstream, not in this repo)**

For each document $x$ in the training corpus, use a strong prompted reasoning model (e.g. GPT-4o / R1) to generate a coarse-to-fine hierarchy:

$$z_T, z_{T-1}, \dots, z_1 \;\sim\; q_{\text{prompted}}(z \mid x)$$

In the current configuration $T = 4$, giving four latent layers $z_4$ (2 words), $z_3$ (4 words), $z_2$ (8 words), $z_1$ (16 words), and the document $x$ (32 words).

**Step 2 — Distil to $q_\phi$ (scripts/03–04)**

Supervised fine-tune a smaller model (Qwen3-4B) on the dataset of $(x,\, z_4, z_3, z_2, z_1)$ tuples. This gives a tractable approximate posterior that imitates the prompted model's latent structure.

**Step 3 — Pre-train $p_\theta$ on $q_\phi$ samples (scripts/05–06)**

Sample latents from $q_\phi$, verify them, and use the verified $(z, x)$ pairs to pre-train `C2FForCausalLM` with a block-prefix causal mask. This gives a well-initialised joint model that already assigns high probability to the kinds of latents $q_\phi$ produces.

### Phase 1 — ELBO Optimisation (script/07)

Starting from the pre-trained $q_\phi$ and $p_\theta$, alternate two gradient steps:

| Phase | Update | Mechanism |
|---|---|---|
| A | $q_\phi$ ($p_\theta$ frozen) | GRPO via veRL; reward = $\log p_\theta(x, z)$ |
| B | $p_\theta$ ($q_\phi$ frozen) | Supervised; sample $z \sim q_\phi$, gradient on $\log p_\theta(x, z)$ |

Freezing one model while updating the other is known to prevent posterior collapse. Since $p_\theta$ has been pre-trained on sensible latents, low-quality (e.g., degenerate or collapsed) latents from $q_\phi$ will receive a low reward signal, giving a natural anti-collapse pressure.

### Sampling from $p_\theta$

1. Sample $z_T$ from $p_\theta(z_T)$ (unconditional generation at coarsest scale).
2. Sample $z_{T-1}$ from $p_\theta(z_{T-1} \mid z_T)$.
3. Continue until $z_1$, then sample $x$ from $p_\theta(x \mid z_T, \dots, z_1)$.

In the current architecture, each conditional is a single forward pass of the C2F transformer — generating one scale per pass, for a total of $T + 1$ passes per document.

---

## Theoretical Background

### Setup

Let $x$ denote a document (32 words in the current configuration). We define the latent sequence

$$z := [z_T, \dots, z_1]$$

where $z_T$ is coarsest and $z_1$ is finest. The two trainable models are:

| Symbol | Role | Implementation |
|---|---|---|
| $q_\phi(z \mid x)$ | Approximate posterior (encoder) | Qwen3-4B fine-tuned by SFT |
| $p_\theta(z, x)$ | Joint generative model (decoder) | `C2FForCausalLM` |

### 1. ELBO Derivation

Starting from the marginal log-likelihood and applying Jensen's inequality:

$$\begin{align}
\log p_\theta(x)
  &= \log \sum_z p_\theta(x, z) \\
  &= \log \sum_z q_\phi(z \mid x)\,\frac{p_\theta(x, z)}{q_\phi(z \mid x)} \\
  &= \log \mathbb{E}_{z \sim q_\phi(z \mid x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \\
  &\geq \mathbb{E}_{z \sim q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \\
  &= \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x, z) - \log q_\phi(z \mid x)\right] \\
  &=: \mathcal{L}(\theta, \phi;\, x)
\end{align}$$

### 2. Gradient with Respect to $\theta$ — Phase B

> This gradient is not computed while $p_\theta$ is frozen.

$$\nabla_\theta \mathcal{L}
= \sum_z q_\phi(z \mid x)\,\nabla_\theta \log p_\theta(x, z)
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[\nabla_\theta \log p_\theta(x, z)\right]$$

Intuitively: push the log-likelihood of the concatenated sequence $(z, x)$ up under $p_\theta$.

**Implementation:** sample $z \sim q_\phi(\cdot \mid x)$ from the frozen SFT model, then take a supervised gradient step on $p_\theta$. This is equivalent to re-running the generation step (step 5) and the C2F pre-training step (step 6) with the current $q_\phi$.

### 3. Gradient with Respect to $\phi$ — Phase A

Define the reward signal

$$s_\phi(z) := \log p_\theta(x, z) - \log q_\phi(z \mid x)$$

so that $\mathcal{L} = \sum_z q_\phi(z \mid x)\,s_\phi(z)$.

Differentiating and applying the product rule:

$$\nabla_\phi \mathcal{L}
= \sum_z \nabla_\phi q_\phi(z \mid x)\,s_\phi(z)
+ \sum_z q_\phi(z \mid x)\,\nabla_\phi s_\phi(z)$$

The second term vanishes because $\nabla_\phi s_\phi(z) = -\nabla_\phi \log q_\phi(z \mid x)$ and $\sum_z q_\phi(z \mid x) = 1$:

$$\sum_z q_\phi(z \mid x)\,\nabla_\phi s_\phi(z)
= -\nabla_\phi \sum_z q_\phi(z \mid x) = 0$$

Applying the REINFORCE identity $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$:

$$\boxed{
\nabla_\phi \mathcal{L}
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[s_\phi(z)\;\nabla_\phi \log q_\phi(z \mid x)\right]
}$$

**Implementation:** this is the GRPO objective. The reward is $\log p_\theta(x, z)$ from a frozen $p_\theta$ forward pass (token-normalised). The $-\log q_\phi(z \mid x)$ KL term is handled by veRL's `actor.use_kl_loss=true`. A format bonus rewards well-formed latent structure.

---

## Model Architecture

The joint model $p_\theta$ is `C2FForCausalLM` (`src/qwen3_joint/`), a Qwen3 transformer with three architectural modifications. Every change from the Qwen3 base is marked with a `# C2F:` comment in the source.

### Sequence Layout

All scales are packed into a single fixed-length sequence:

```
Position:  0       1–2        3–6        7–14       15–30      31–62     63
Token:    [BOS | z_4 (×2) | z_3 (×4) | z_2 (×8) | z_1 (×16) | x (×32) | pad]
Scale:      —      0          1          2           3           4
```

Total content: 63 tokens → padded to 64 (nearest power of two). Controlled by `scale_lengths: [2, 4, 8, 16, 32]` in the config.

### Block-Prefix Causal Attention Mask

Token at scale $k$ attends to all tokens at strictly coarser scales ($< k$), but not to tokens within the same scale. BOS is a universal prefix attended by every token.

```
z_4 tokens (scale 0)  →  attend to BOS only
z_3 tokens (scale 1)  →  attend to BOS + z_4
z_2 tokens (scale 2)  →  attend to BOS + z_4 + z_3
z_1 tokens (scale 3)  →  attend to BOS + z_4 + z_3 + z_2
 x  tokens (scale 4)  →  attend to BOS + z_4 + z_3 + z_2 + z_1
```

Flash Attention 2 is explicitly disabled (`attn_implementation="eager"`) because FA2 ignores custom additive masks, which would silently break this pattern.

### Per-Scale Position Embeddings (`C2FScaleEmbedding`)

Replaces Qwen3's Rotary Position Embeddings (RoPE). Each scale has its own learned absolute embedding table `nn.Embedding(scale_length_k, hidden_size)`, plus a separate learned BOS vector. Embeddings are added to token embeddings before the transformer layers.

### Unshifted Cross-Entropy Loss

`logits[i]` predicts `labels[i]` (not `labels[i+1]` as in standard autoregressive LMs). The block-prefix mask determines what context each position can see; within-scale tokens are predicted independently conditioned on all coarser scales.

### Space Tokenizer

A word-level tokenizer (one space-separated word = one token) is trained from the dataset vocabulary and saved in HuggingFace format (`checkpoints/decoder/tokenizer/`). This gives a strict 1:1 word-to-token mapping so scale boundaries are always at deterministic positions for verified inputs — no truncation or intra-scale padding is possible.

---

## Pipeline Overview

Steps 1–2 run upstream (not in this repo). Steps 3–7 correspond to files in `scripts/`.

```
Steps 1–2 (upstream)
  Corpus collection → Batch API latent generation (z_n: format)
  → data/batch_outputs/

Step 3   scripts/03_verify_outputs.py
  Verify z_n: format and exact word counts
  → data/sft_dataset/train.parquet    [columns: prompt, response]

Step 4   scripts/04_sft_train.py
  SFT of Qwen3-4B as q_φ encoder (veRL + torchrun)
  → checkpoints/sft/

Step 5   scripts/05_generate_local.py
  vLLM generates z ~ q_φ(z|x) at scale, verify, flatten
  → data/local_generations/c2f_train.parquet

Step 6   scripts/06_train_decoder.py
  Pretrain C2FForCausalLM as p_θ joint model (HF Trainer + FSDP)
  → checkpoints/decoder/

Step 7   scripts/07_rl_train.py
  ELBO optimisation — alternating GRPO + supervised fine-tuning
  Phase A: GRPO on q_φ, p_θ frozen    → checkpoints/rl/sft/
  Phase B: supervised p_θ, q_φ frozen → checkpoints/rl/c2f/
```

| Step | Script | Key inputs | Output |
|---|---|---|---|
| 3 | `03_verify_outputs.py` | `batch_outputs/*.jsonl`, `sft.jsonl` | `data/sft_dataset/train.parquet` |
| 4 | `04_sft_train.py` | `sft_dataset/train.parquet` | `checkpoints/sft/` |
| 5 | `05_generate_local.py` | `checkpoints/sft/`, prompts | `data/local_generations/c2f_train.parquet` |
| 6 | `06_train_decoder.py` | `c2f_train.parquet` or `train.parquet` | `checkpoints/decoder/` |
| 7A | `07_rl_train.py --phase sft` | `checkpoints/sft/`, `checkpoints/decoder/` | `checkpoints/rl/sft/` |
| 7B | `07_rl_train.py --phase c2f` | `checkpoints/rl/sft/`, `checkpoints/decoder/` | `checkpoints/rl/c2f/` |

---

## Setup

### Prerequisites

- Python ≥ 3.12, [`uv`](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU required for steps 4–7 (1–4 × A100 / H100 / H200 recommended)

### Installation

Each pipeline step has its own optional dependency group to avoid unnecessary installs:

```bash
pip install -e ".[sft]"        # Step 4 — veRL SFT
pip install -e ".[generation]" # Step 5 — vLLM generation
pip install -e ".[c2f]"        # Step 6 — C2F pretraining
pip install -e ".[rl]"         # Step 7 — RL ELBO training
```

### Secrets

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

```env
# Weights & Biases experiment tracking
WANDB_API_KEY=your-key-here
WANDB_PROJECT=coarse-to-fine
WANDB_ENTITY=your-entity

# Batch API — steps 1–2 (upstream)
OPENAI_API_KEY=sk-...

# Gated HF models (Qwen3)
HF_TOKEN=hf_...
```

`.env` is gitignored and never committed. On HPC, export these variables in your job script — existing environment variables take precedence over `.env` values.

---

## Running the Pipeline

### Step 3 — Verify Batch Outputs

Parses batch API response JSON lines, checks each output against word-count constraints ($z_4$: 2, $z_3$: 4, $z_2$: 8, $z_1$: 16 words), and writes the SFT parquet.

```bash
python scripts/03_verify_outputs.py \
  --input  data/batch_outputs/.../output.jsonl \
  --config config/experiments/latent_generation.yaml \
  --prompts data/prompt_data/.../sft.jsonl \
  --output data/verified/latent_generation_10k_v1
```

Output: `data/sft_dataset/train.parquet` with columns `prompt` (32-word document) and `response` (raw `z_4: ...\nz_3: ...\nz_2: ...\nz_1: ...`).

---

### Step 4 — Supervised Fine-Tuning

Fine-tunes Qwen3-4B as $q_\phi$ on the verified parquet using veRL's FSDP SFT trainer.

```bash
python scripts/04_sft_train.py --config config/experiments/latent_generation.yaml

# Pass Hydra overrides directly to veRL:
python scripts/04_sft_train.py \
  trainer.total_epochs=1 \
  data.micro_batch_size_per_gpu=16
```

GPU count is set by `sft.num_gpus` in the config (1–4). SLURM: `scripts/slurm_04_sft.sh`.

---

### Step 5 — Generate Latents with vLLM

Runs the SFT model at scale using vLLM offline inference, verifies word counts, and flattens outputs for C2F training.

```bash
python scripts/05_generate_local.py --config config/experiments/latent_generation.yaml

# Limit samples or change output directory:
python scripts/05_generate_local.py \
  --config config/experiments/latent_generation.yaml \
  --num-samples 100000 \
  --output-dir data/local_generations/run2
```

Outputs:
- `data/local_generations/generations.parquet` — raw model outputs
- `data/local_generations/c2f_train.parquet` — flattened word sequences ready for step 6

SLURM: `scripts/slurm_05_generate.sh`.

---

### Step 6 — Pretrain C2F Joint Model

Trains $p_\theta$ (`C2FForCausalLM`) on the flattened latent+text sequences using HF Trainer with optional FSDP.

```bash
# Single GPU:
python scripts/06_train_decoder.py --config config/experiments/latent_generation.yaml

# Multi-GPU with FSDP:
accelerate launch --num_processes=4 scripts/06_train_decoder.py \
  --config config/experiments/latent_generation.yaml

# Resume from a checkpoint:
python scripts/06_train_decoder.py \
  --config config/experiments/latent_generation.yaml \
  --resume-from checkpoints/decoder/checkpoint-500
```

The `c2f_training.dataset_format` config key controls the input format:
- `"c2f"` — flat word sequences from step 5 (`c2f_train.parquet`)
- `"sft"` — prompt+response from step 3 (`train.parquet`), flattened on the fly

SLURM: `scripts/slurm_06_pretrain.sh`.

---

### Step 7 — ELBO Optimisation

Alternates Phase A (GRPO on $q_\phi$) and Phase B (supervised on $p_\theta$). Each phase can be run independently or both sequentially in one call.

```bash
# Phase A only — GRPO on q_φ (C2F frozen):
python scripts/07_rl_train.py --phase sft \
  --config config/experiments/latent_generation.yaml

# Phase B only — supervised p_θ (SFT frozen):
python scripts/07_rl_train.py --phase c2f \
  --config config/experiments/latent_generation.yaml

# One full round of alternation (Phase A then Phase B):
python scripts/07_rl_train.py --phase both \
  --config config/experiments/latent_generation.yaml
```

Dot-path overrides (e.g. `rl.sft_rl.epochs=1`) update the in-memory config and apply to the corresponding phase. Non-`rl.*` overrides are forwarded as Hydra overrides directly to the veRL trainer.

```bash
# Smoke-test Phase A (small batch):
python scripts/07_rl_train.py --phase sft \
  --config config/experiments/latent_generation.yaml \
  rl.sft_rl.epochs=1 rl.sft_rl.train_batch_size=8 rl.sft_rl.rollout_n=4

# Smoke-test Phase B (100 prompts, 1 epoch):
python scripts/07_rl_train.py --phase c2f \
  --config config/experiments/latent_generation.yaml \
  rl.c2f_finetune.num_samples=100 rl.c2f_finetune.epochs=1
```

SLURM: `scripts/slurm_07_rl.sh`. Set `PHASE=sft|c2f|both` before submitting.

---

## Configuration

All parameters live in `config/experiments/latent_generation.yaml`. Key sections:

| Section | Controls |
|---|---|
| `scale_lengths` | Token positions per scale `[2, 4, 8, 16, 32]`; determines `seq_len = 64` |
| `word_count_constraints` | Exact word counts per latent layer (used by verifier and reward function) |
| `wandb` | W&B enable/disable, project, entity, group, tags, mode |
| `sft` | Step 4: base model name, GPU count, batch size, LR, epochs |
| `generation` | Step 5: SFT checkpoint path, vLLM tensor parallelism, sampling params |
| `c2f_training` | Step 6: init source, tokenizer type, FSDP config, training hyperparams |
| `rl.sft_rl` | Step 7A: GRPO rollout group size, KL coefficient, format bonus weight |
| `rl.c2f_finetune` | Step 7B: generation settings and C2F fine-tuning hyperparams |

Key `rl.sft_rl` fields:

| Field | Default | Description |
|---|---|---|
| `rollout_n` | 8 | GRPO group size (samples per prompt) |
| `kl_coef` | 0.01 | KL penalty weight in actor loss |
| `format_bonus_weight` | 0.1 | Added to reward when all latent layers have the correct word count |
| `temperature` | 1.0 | Sampling temperature during rollout |

---

## Project Layout

```
coarse-to-fine/
├── config/
│   └── experiments/
│       └── latent_generation.yaml   # single config file for all steps
│
├── scripts/
│   ├── 03_verify_outputs.py         # step 3 — verify + build SFT parquet
│   ├── 04_sft_train.py              # step 4 — veRL SFT via torchrun
│   ├── 05_generate_local.py         # step 5 — vLLM generation + flatten
│   ├── 06_train_decoder.py          # step 6 — C2F pretraining via HF Trainer
│   ├── 07_rl_train.py               # step 7 — ELBO optimisation (GRPO + SFT)
│   └── slurm_*.sh                   # SLURM job scripts for steps 4–7
│
├── src/
│   ├── batch/           step 2 — batch API client and polling
│   ├── verification/    step 3 — RuleBasedVerifier, word-count checks
│   ├── sft/             step 4 — veRL Hydra override builder
│   ├── generation/      step 5 — LatentGenerator (vLLM), flatten_for_c2f
│   ├── qwen3_joint/     C2F model architecture — C2FConfig, C2FForCausalLM
│   ├── c2f_training/    step 6 — C2FDataset, C2FTrainer, space tokenizer
│   ├── rl/              step 7 — C2FRewardManager, verl_config, train
│   ├── data/            Pydantic schemas, sharding, utilities
│   └── utils/           load_env, setup_wandb, logging, io
│
├── data/
│   ├── batch_outputs/           raw batch API responses
│   ├── sft_dataset/             step 3 output  (train.parquet)
│   ├── local_generations/       step 5 output  (c2f_train.parquet)
│   └── rl_dataset/              step 7 data    (sft_rl.parquet, c2f_finetune/)
│
├── checkpoints/
│   ├── sft/                     q_φ SFT checkpoint (step 4)
│   ├── decoder/                 p_θ C2F checkpoint (step 6)
│   └── rl/
│       ├── sft/                 q_φ after Phase A GRPO
│       └── c2f/                 p_θ after Phase B supervised fine-tuning
│
├── pyproject.toml               optional-dep groups: sft, generation, c2f, rl
└── .env.example                 template for WANDB_API_KEY, OPENAI_API_KEY, HF_TOKEN
```

---

## Monitoring with W&B

Enable W&B by setting `wandb.enabled: true` in the experiment YAML and providing `WANDB_API_KEY` in `.env`.

| Step | Key metrics |
|---|---|
| 4 — SFT | `train/loss`, `eval/loss`, learning rate schedule |
| 5 — Generation | Verification pass rate (% of outputs meeting word-count constraints) |
| 6 — C2F pretrain | `loss_z_4`, `loss_z_3`, `loss_z_2`, `loss_z_1`, `loss_text` (per-scale CE loss) |
| 7A — GRPO ($q_\phi$) | Token-normalised reward ($\log p_\theta$) trending up; format pass rate increasing; KL stable |
| 7B — C2F SFT ($p_\theta$) | `eval/loss` decreasing on freshly generated $q_\phi$ samples |

Each step adds a run tag (`sft`, `generation`, `c2f-pretrain`, `rl-sft`, `rl-c2f`) so runs are filterable in the W&B dashboard. All scripts call `load_env()` and `setup_wandb()` from `src/utils/env.py` before any training-framework imports.
