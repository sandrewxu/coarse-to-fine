# Coarse-to-Fine Latent Language Model

Most discrete diffusion approaches use a **fixed** forward process — independently masking tokens, adding Gaussian noise in embedding space, or applying some other corruption schedule that has no semantic meaning. The reverse process must then "undo" corruption steps that were never designed to be meaningful.

This project takes a different route: **learn** the forward and reverse processes jointly by optimising the ELBO, as in a variational autoencoder, while making use of the strong natural-language prior of a pretrained autoregressive LM. The result is a coarse-to-fine generative process that first produces a high-level document summary $z_T$, then a slightly more detailed one $z_{T-1}$, and so on until generating a coherent document $x$. Because the hierarchy is learned rather than fixed, each latent level has a well-defined semantic interpretation.

## Quick start

```bash
git clone <this-repo> && cd coarse-to-fine
make install                              # uv sync --extra dev + pre-commit hooks
cp .env.example .env && $EDITOR .env      # fill in WANDB / OpenAI / HF tokens
make test                                 # smoke check the unit suite
```

For the full pipeline, jump to [Running the Pipeline](#running-the-pipeline).
For agent / Claude Code orientation, see [`CLAUDE.md`](./CLAUDE.md).

## Current status

The core algorithm is implemented end-to-end (steps 0–9) and two baselines are
trained with matched architecture for fair comparison:

- **C2F decoder** (`scripts/06_train_decoder.py`) — supports both `block` and
  `causal` masks.
- **AR baseline** (`scripts/06b_train_ar_baseline.py`) — stock
  `Qwen3ForCausalLM` on text only, same space tokenizer + 95/5 split.
- **MDLM diffusion baseline** (`scripts/06c_train_diffusion_baseline.py`) —
  continuous-time SUBS masked-diffusion head on the same backbone.
- **Joint ELBO training** (`scripts/07_rl_train.py --phase joint`) — RL
  posterior and online MLE decoder updates in a single veRL reward loop.
- **Unified NLL evaluator** (`scripts/09_eval_nll.py`) — exact AR NLL, C2F
  IWAE-1 ELBO, C2F ELBO upper bound via the SFT posterior, and MDLM MC-NELBO,
  all reported in matched nats/word with bootstrap 95 % CI.

HPC configs for H100 (80 GB) and H200 (141 GB) live alongside the base YAML in
`config/`; both mask variants (`block`, `causal`) are supported in the joint
phase via `rl.joint.c2f_mask_type`.

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

- **Block-prefix causal attention**: the joint model $p_\theta$ uses a transformer with a custom attention mask so that tokens at scale $k$ attend to all coarser scales but are independent within the same scale. This is in the spirit of [VAR](https://arxiv.org/abs/2404.02905) and [Block Diffusion](https://arxiv.org/abs/2503.09573). A standard **causal** mask mode is also supported for comparison.
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

**Step 1 — Synthetic latent generation (scripts/00–02)**

For each document $x$ in the training corpus, use a strong prompted reasoning model (e.g. GPT-4o / R1) via the OpenAI Batch API to generate a coarse-to-fine hierarchy:

$$z_T, z_{T-1}, \dots, z_1 \;\sim\; q_{\text{prompted}}(z \mid x)$$

In the current configuration $T = 4$, giving four latent layers $z_4$ (2 words), $z_3$ (4 words), $z_2$ (8 words), $z_1$ (16 words), and the document $x$ (32 words).

**Step 2 — Distil to $q_\phi$ (scripts/03–04)**

Supervised fine-tune a smaller model (Qwen3-4B) on the dataset of $(x,\, z_4, z_3, z_2, z_1)$ tuples using HuggingFace Trainer. This gives a tractable approximate posterior that imitates the prompted model's latent structure.

**Step 3 — Pre-train $p_\theta$ on $q_\phi$ samples (scripts/05–06)**

Sample latents from $q_\phi$, verify them, and use the verified $(z, x)$ pairs to pre-train `C2FForCausalLM` with a block-prefix causal mask (or standard causal mask). This gives a well-initialised joint model that already assigns high probability to the kinds of latents $q_\phi$ produces.

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

The joint model $p_\theta$ is `C2FForCausalLM` (`src/c2f_model/`), a Qwen3 transformer with three architectural modifications. Every change from the Qwen3 base is marked with a `# C2F:` comment in the source.

### Sequence Layout

All scales are packed into a single fixed-length sequence:

```
Position:  0       1–2        3–6        7–14       15–30      31–62     63
Token:    [BOS | z_4 (×2) | z_3 (×4) | z_2 (×8) | z_1 (×16) | x (×32) | pad]
Scale:      —      0          1          2           3           4
```

Total content: 63 tokens → padded to 64 (nearest power of two). Controlled by `scale_lengths: [2, 4, 8, 16, 32]` in the config.

### Attention Mask

`C2FForCausalLM` supports two attention mask types, selected via `mask_type` in the config:

**Block-prefix (`mask_type: "block"`, default):** Token at scale $k$ attends to all tokens at strictly coarser scales ($< k$), but not to tokens within the same scale. BOS is a universal prefix attended by every token.

```
z_4 tokens (scale 0)  →  attend to BOS only
z_3 tokens (scale 1)  →  attend to BOS + z_4
z_2 tokens (scale 2)  →  attend to BOS + z_4 + z_3
z_1 tokens (scale 3)  →  attend to BOS + z_4 + z_3 + z_2
 x  tokens (scale 4)  →  attend to BOS + z_4 + z_3 + z_2 + z_1
```

**Causal (`mask_type: "causal"`):** Standard lower-triangular autoregressive mask where each token attends to all preceding tokens. Useful as a baseline comparison.

Flash Attention 2 is explicitly disabled (`attn_implementation="eager"`) because FA2 ignores custom additive masks, which would silently break the block-prefix pattern.

### Per-Scale Position Embeddings (`C2FScaleEmbedding`)

Replaces Qwen3's Rotary Position Embeddings (RoPE). Each scale has its own learned absolute embedding table `nn.Embedding(scale_length_k, hidden_size)`, plus a separate learned BOS vector. Embeddings are added to token embeddings before the transformer layers.

### Unshifted Cross-Entropy Loss

`logits[i]` predicts `labels[i]` (not `labels[i+1]` as in standard autoregressive LMs). The block-prefix mask determines what context each position can see; within-scale tokens are predicted independently conditioned on all coarser scales.

### Space Tokenizer

A word-level tokenizer (one space-separated word = one token) is trained from the dataset vocabulary and saved in HuggingFace format (`checkpoints/decoder/tokenizer/`). This gives a strict 1:1 word-to-token mapping so scale boundaries are always at deterministic positions for verified inputs — no truncation or intra-scale padding is possible.

---

## Pipeline Overview

All steps (0–7) correspond to numbered scripts in `scripts/`.

```
Step 0   scripts/00_prepare_data.py
  Download HF dataset, preprocess, shuffle, split into chunks
  → data/tinystoriesv2_shuffled/
      tinystoriesv2.chunk.{00-07}.jsonl   (8 training chunks, ~3M docs each)
      tinystoriesv2.prompt.jsonl           (200k docs for batch API)
      tinystoriesv2.val.jsonl              (40k validation)
      tinystoriesv2.test.jsonl             (40k test)
      tinystoriesv2.rl.jsonl               (64k for RL training)

Step 1   scripts/01_create_batch_requests.py
  Format prompt-split documents into OpenAI Batch API request JSONL
  → data/prompt_data/{model}/docs_{N}/{user_prompt}/{system_prompt}/sft.jsonl

Step 2   scripts/02_submit_batch.py
  Submit batch request, monitor, and download results
  → data/batch_outputs/{metadata}/{batch_id}/output.jsonl

Step 3   scripts/03_verify_outputs.py
  Verify z_n: format and exact word counts
  → data/sft_dataset/train.parquet    [columns: prompt, response]

Step 4   scripts/04_sft_train.py
  SFT of Qwen3-4B as q_φ encoder on batch API data (HF Trainer ± FSDP)
  → checkpoints/sft/

Step 5   scripts/05_generate_local.py
  Generate z ~ q_φ(z|x) from chunk files via vLLM, verify, flatten
  → data/local_generations/c2f_train.parquet

Step 6   scripts/06_train_decoder.py
  Pretrain C2FForCausalLM as p_θ on step 5 output (HF Trainer + FSDP)
  Supports --mask-type block (default) or causal
  → checkpoints/decoder/

Step 6b  scripts/06b_train_ar_baseline.py
  No-latents AR baseline (stock Qwen3ForCausalLM). Matches c2f_training's
  architecture, tokenizer, and 95/5 split for apples-to-apples NLL.
  → checkpoints/ar_baseline/

Step 6c  scripts/06c_train_diffusion_baseline.py
  MDLM continuous-time SUBS masked-diffusion baseline. Same Qwen3 backbone,
  same space tokenizer ([MASK] is a dedicated reserved id).
  → checkpoints/diffusion_baseline/

Step 7   scripts/07_rl_train.py
  ELBO optimisation on RL split — alternating GRPO + supervised
  Phase A (sft):    GRPO on q_φ, p_θ frozen              → checkpoints/rl/sft/
  Phase B (c2f):    supervised p_θ, q_φ frozen           → checkpoints/rl/c2f/
  Phase (joint):    REINFORCE on q_φ + online MLE on p_θ → checkpoints/rl/joint/

Step 8   scripts/08_eval_joint.py
  Posterior-collapse diagnostic: generate z sequences from p_θ in causal
  mode and report unique-token ratio + top-5 most common z tokens.

Step 9   scripts/09_eval_nll.py
  Unified NLL evaluator. Four code paths under one CLI, all reporting
  nats/word + bootstrap 95 % CI:
    ar         exact shifted-CE NLL on raw text
    c2f        IWAE-1 ELBO lower bound with gold z from the parquet
    c2f_bound  ELBO upper bound on -log p(x)/T via the SFT model as q_φ
               (enabled with --sft-ckpt; directly comparable to AR)
    diffusion  MC-NELBO (N samples of t, mask) — matches the 06c training loss
```

| Step | Script | Required args | Key inputs | Output |
|---|---|---|---|---|
| 0 | `00_prepare_data.py` | `--dataset` | HuggingFace dataset | `data/{dataset}_shuffled/` |
| 1 | `01_create_batch_requests.py` | `--config` | `dataset.prompt_split` | `data/prompt_data/.../sft.jsonl` |
| 2 | `02_submit_batch.py` | `--input` | Step 1 sft.jsonl | `data/batch_outputs/.../output.jsonl` |
| 3 | `03_verify_outputs.py` | `--input`, `--config` | `output.jsonl`, `sft.jsonl` | `data/sft_dataset/train.parquet` |
| 4 | `04_sft_train.py` | `--data` | `train.parquet` (batch API data) | `checkpoints/sft/` |
| 5 | `05_generate_local.py` | `--chunks`, `--config` | Chunk JSONL files, SFT checkpoint | `data/local_generations/c2f_train.parquet` |
| 6 | `06_train_decoder.py` | `--data` | `c2f_train.parquet` (step 5 output) | `checkpoints/decoder/` |
| 6b | `06b_train_ar_baseline.py` | `--data` | `c2f_train.parquet` (shared with step 6) | `checkpoints/ar_baseline/` |
| 6c | `06c_train_diffusion_baseline.py` | `--data` | `c2f_train.parquet` (shared with step 6) | `checkpoints/diffusion_baseline/` |
| 7 | `07_rl_train.py` | `--phase`, `--config` | SFT + C2F checkpoints, RL split | `checkpoints/rl/{sft,c2f,joint}/` |
| 8 | `08_eval_joint.py` | `--checkpoint` | C2F checkpoint | Posterior-collapse diagnostics (stdout) |
| 9 | `09_eval_nll.py` | `--model-kind`, `--ckpt`, `--test` | Checkpoint + held-out data | Aggregate nats/word + per-scale breakdown |

---

## Setup

### Prerequisites

- Python ≥ 3.12, [`uv`](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU required for steps 4–7 (1–4 × A100 / H100 / H200 recommended)

### Installation

Each pipeline step has its own optional dependency group to avoid unnecessary installs:

```bash
pip install -e ".[sft]"        # Step 4 — HF Trainer (accelerate + torch) for q_φ SFT
pip install -e ".[generation]" # Step 5 — vLLM inference for q_φ sampling
pip install -e ".[c2f]"        # Steps 6 / 6b / 6c — HF Trainer + FSDP (+ liger-kernel)
pip install -e ".[rl]"         # Step 7 — veRL (+ flash-attn; needs --no-build-isolation)
pip install -e ".[dev]"        # Ruff, pytest, pre-commit, nbstripout, CPU torch (for unit tests)
```

For **contributor setup** (lint/format hooks):

```bash
make install     # uv sync --extra dev + pre-commit install
```

### HPC build note (flash-attn / vllm)

On Yale HPC (H100), `flash-attn` and `vllm` wheels often need to be rebuilt from
source against the cluster's CUDA/GCC toolchain. From an interactive GPU session:

```bash
ml load CUDA/12.8.0 GCC/13.3.0
export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=12 NVCC_THREADS=2
export UV_PREFER_BINARY=1
uv sync --extra generation --no-build-isolation
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

# Batch API — steps 1–2
OPENAI_API_KEY=sk-...

# Gated HF models (Qwen3)
HF_TOKEN=hf_...
```

`.env` is gitignored and never committed. On HPC, export these variables in your job script — existing environment variables take precedence over `.env` values.

---

## Running the Pipeline

### Step 0 — Prepare Data

Downloads a HuggingFace dataset, preprocesses it (for TinyStories: splits on `<|endoftext|>` and chunks into 32-word documents), shuffles with [terashuf](https://github.com/alexandres/terashuf), and splits into sharded training chunks plus dedicated validation, test, prompt, and RL splits.

```bash
# Default: TinyStories → 4 chunks + val/test/prompt/rl splits
python scripts/00_prepare_data.py --dataset tinystoriesv2 --memory 8

# With config for split sizes and seed:
python scripts/00_prepare_data.py \
  --dataset tinystoriesv2 \
  --config config/latent_generation.yaml

# Different dataset:
python scripts/00_prepare_data.py --dataset fineweb_edu_10bt --memory 32 --data-dir /scratch/data
```

Output structure (default: `data/tinystoriesv2_shuffled/`):

| File | Purpose | Size (default) |
|---|---|---|
| `tinystoriesv2.chunk.{00-07}.jsonl` | Training data shards (for steps 5–6) | ~3M docs each |
| `tinystoriesv2.prompt.jsonl` | Documents for batch API generation (step 1) | 200k (8 × 25k) |
| `tinystoriesv2.val.jsonl` | Validation set | 40k (8 × 5k) |
| `tinystoriesv2.test.jsonl` | Test set | 40k (8 × 5k) |
| `tinystoriesv2.rl.jsonl` | Documents for RL training (step 7) | 64k (8 × 8k) |

Split sizes are configurable via the `data_prep` config section (`k_prompt`, `k_validation`, `k_test`, `k_rl`).

Supported datasets: `tinystoriesv2`, `fineweb_edu`, `fineweb_edu_10bt`, `dclm_baseline_1.0`, `dclm_baseline_1.0_10prct`, `dclm_pool_1b_1x`, `cosmopedia_v2`, `python_edu` (see `src/data/registry.py`).

SLURM: `scripts/slurm_00_prepare_data.sh`. Override dataset with `DATASET=fineweb_edu_10bt`.

---

### Step 1 — Create Batch API Requests

Formats documents from the prompt split into OpenAI Batch API request JSONL files. Each request includes a system prompt, few-shot examples, and the document text.

```bash
# Derive docs path from config (dataset.data_dir / dataset.prompt_split):
python scripts/01_create_batch_requests.py \
  --config config/latent_generation.yaml

# Explicit docs path:
python scripts/01_create_batch_requests.py \
  --docs data/tinystoriesv2_shuffled/tinystoriesv2.prompt.jsonl \
  --config config/latent_generation.yaml

# Process a subset (last 10k documents):
python scripts/01_create_batch_requests.py \
  --config config/latent_generation.yaml \
  --doc-start -10000
```

The output path is printed and encodes the model, doc count, and prompt names:
```
data/prompt_data/gpt-5-nano-2025-08-07/docs_0000100000/gemini-3-pro-7/latent-generation/sft.jsonl
```

**Important:** After running this step, update `sft.prompt_data` in `config/latent_generation.yaml` to the output path. Step 3 uses this to extract original prompts during verification.

Prompt templates are stored in `prompts/` at the project root:
- `prompts/system_prompts/latent-generation.txt` — instructions for latent hierarchy generation
- `prompts/user_prompts/gemini-3-pro-7.txt` — user message template with `{doc}` placeholder
- `prompts/few_shot_examples/latent-generation.jsonl` — 7 example (user, assistant) pairs

SLURM: `scripts/slurm_01_batch_create.sh`.

---

### Step 2 — Submit and Monitor Batch

Submits the request JSONL to the OpenAI Batch API, optionally polls until completion, and downloads the results.

```bash
# Submit and monitor until done:
python scripts/02_submit_batch.py \
  --input data/prompt_data/.../sft.jsonl \
  --config config/latent_generation.yaml \
  --monitor

# Submit only (check back later):
python scripts/02_submit_batch.py \
  --input data/prompt_data/.../sft.jsonl \
  --config config/latent_generation.yaml

# Download all completed batches filtered by run tag:
python scripts/02_submit_batch.py \
  --download \
  --run-tag latent_generation_10k_v1

# Monitor all active batches:
python scripts/02_submit_batch.py --monitor-all
```

Requires `OPENAI_API_KEY` in `.env`. Results are saved to `data/batch_outputs/` with metadata encoded in the directory structure.

SLURM: `scripts/slurm_02_batch_submit.sh`. Set `INPUT_FILE` to the sft.jsonl from step 1.

---

### Step 3 — Verify Batch Outputs

Parses batch API response JSON lines, checks each output against word-count constraints ($z_4$: 2, $z_3$: 4, $z_2$: 8, $z_1$: 16 words), and writes the SFT parquet.

```bash
python scripts/03_verify_outputs.py \
  --input  data/batch_outputs/.../output.jsonl \
  --config config/latent_generation.yaml \
  --prompts data/prompt_data/.../sft.jsonl \
  --output data/verified/latent_generation_10k_v1
```

Output: `data/sft_dataset/train.parquet` with columns `prompt` (32-word document) and `response` (raw `z_4: ...\nz_3: ...\nz_2: ...\nz_1: ...`).

---

### Step 4 — Supervised Fine-Tuning

Fine-tunes Qwen3-4B as $q_\phi$ on the verified parquet using HuggingFace Trainer. Trains the model to map documents to latent hierarchies using the chat format. Saves standard HuggingFace checkpoints (`model.safetensors`) that vLLM and downstream steps can load directly.

```bash
# Minimal — just point at data:
python scripts/04_sft_train.py --data data/sft_dataset/train.parquet

# With config for model/lr/batch/W&B defaults:
python scripts/04_sft_train.py \
  --data data/sft_dataset/train.parquet \
  --config config/latent_generation.yaml

# Override training hyperparams:
python scripts/04_sft_train.py \
  --data data/sft_dataset/train.parquet \
  --num-gpus 2 --epochs 3

# Multi-GPU with FSDP:
accelerate launch --num_processes=2 scripts/04_sft_train.py \
  --data data/sft_dataset/train.parquet \
  --config config/latent_generation.yaml
```

SLURM: `scripts/slurm_04_sft.sh`.

---

### Step 5 — Generate Latents

Runs the trained $q_\phi$ model on raw documents from **chunk files** to generate latent hierarchies. The `--chunks` argument selects which chunk indices to use (default: `[0, 1, 2, 3]` from config). Outputs are verified and flattened for C2F training.

```bash
# Generate from chunks 0-3 (default), using config for model path and sampling:
python scripts/05_generate_local.py \
  --chunks 0 1 2 3 \
  --config config/latent_generation.yaml

# Use all chunks for config defaults:
python scripts/05_generate_local.py \
  --config config/latent_generation.yaml

# Subset of samples for testing:
python scripts/05_generate_local.py \
  --chunks 0 \
  --config config/latent_generation.yaml \
  --num-samples 1000

# Backward compat — generate from a parquet file:
python scripts/05_generate_local.py \
  --data data/sft_dataset/train.parquet \
  --model checkpoints/sft/checkpoint-1172 \
  --output-dir data/local_generations
```

Outputs: `generations.parquet` (raw) and `c2f_train.parquet` (flattened for C2F training, if config provided).

SLURM: `scripts/slurm_05_generate.sh`. Override chunks with `CHUNKS="0 1"`.

---

### Step 6 — Pretrain C2F Joint Model

Trains $p_\theta$ (`C2FForCausalLM`) on the **step 5 output** (`c2f_train.parquet`) using HF Trainer with optional FSDP. Dataset format is auto-detected from parquet columns (`text` = c2f, `prompt`+`response` = sft). Supports two attention mask modes: block-prefix (default) and standard causal.

```bash
# Train on step 5 output with block-prefix mask (default):
python scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/latent_generation.yaml

# Train with standard causal attention mask:
python scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/latent_generation.yaml \
  --mask-type causal

# Init from pretrained weights:
python scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --init-from Qwen/Qwen3-4B \
  --config config/latent_generation.yaml

# Override training hyperparams from CLI:
python scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --epochs 5 --lr 1e-4 --batch-size 16

# Multi-GPU with FSDP:
accelerate launch --num_processes=4 scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/latent_generation.yaml

# Resume from checkpoint:
python scripts/06_train_decoder.py \
  --data data/local_generations/c2f_train.parquet \
  --resume-from checkpoints/decoder/checkpoint-500
```

SLURM: `scripts/slurm_06_pretrain.sh`.

---

### Step 6b — AR Baseline (no latents)

Stock `Qwen3ForCausalLM` trained on the **same parquet** as step 6 but reading
only the document text. Architecture overrides live under `ar_training:` in the
YAML and match `c2f_training:` so `09_eval_nll.py` can compare nats/word
directly.

```bash
python scripts/06b_train_ar_baseline.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/latent_generation.yaml

# Multi-GPU with FSDP:
accelerate launch --num_processes=4 scripts/06b_train_ar_baseline.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/latent_generation.yaml
```

The space tokenizer (produced by step 6) is reused for vocabulary parity; the
tokenizer is saved alongside the AR checkpoint.

SLURM: `scripts/slurm_06b_ar_baseline.sh`.

---

### Step 6c — MDLM Diffusion Baseline

Continuous-time SUBS masked-diffusion head (Sahoo et al. 2024) on the same
Qwen3 backbone, same parquet, same space tokenizer. The tokenizer reserves a
dedicated `[MASK]` token so `x0 == mask_id` is impossible by construction and
vocab parity with AR / C2F is preserved.

```bash
python scripts/06c_train_diffusion_baseline.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/H100_joint_causal.yaml

accelerate launch --num_processes=4 scripts/06c_train_diffusion_baseline.py \
  --data data/local_generations/c2f_train.parquet \
  --config config/H100_joint_causal.yaml
```

Math lives in `src/c2f_model/training/diffusion.py`; the same `mdlm_loss`
function powers the training step and the evaluator in `src/eval/diffusion.py`,
so training and eval cannot drift.

SLURM: `scripts/slurm_06c_diffusion_baseline.sh`.

---

### Step 7 — ELBO Optimisation

Alternates Phase A (GRPO on $q_\phi$) and Phase B (supervised on $p_\theta$), using the **RL split** (`dataset.rl_split`) as the document pool — a separate held-out set from the data used in steps 5–6. Each phase can be run independently or both sequentially.

```bash
# Phase A only — GRPO on q_φ (C2F frozen):
python scripts/07_rl_train.py --phase sft --config config/latent_generation.yaml

# Phase B only — supervised p_θ (SFT frozen):
python scripts/07_rl_train.py --phase c2f --config config/latent_generation.yaml

# One full round (Phase A then Phase B):
python scripts/07_rl_train.py --phase both --config config/latent_generation.yaml

# Joint training — REINFORCE on q_φ with online MLE updates on p_θ:
python scripts/07_rl_train.py --phase joint --config config/latent_generation.yaml
```

The joint phase uses `JointC2FRewardManager` (`src/rl/reward_joint.py`) to run a trainable $p_\theta$ inside the reward loop: per rollout sample, $p_\theta$ takes an MLE gradient step on $(z, x)$ and the `-CE_loss` is returned as the reward. $q_\phi$ is updated with REINFORCE via veRL. Malformed rollouts receive a negative reward (`rl.joint.malformed_reward`). Checkpoints are saved every `c2f_save_steps` under `checkpoints/rl/joint/c2f/`.

Dot-path overrides (e.g. `rl.sft_rl.epochs=1`) update the in-memory config. Non-`rl.*` overrides are forwarded as Hydra overrides to the veRL trainer.

```bash
# Smoke-test Phase A:
python scripts/07_rl_train.py --phase sft --config config/latent_generation.yaml \
  rl.sft_rl.epochs=1 rl.sft_rl.train_batch_size=8

# Smoke-test Phase B:
python scripts/07_rl_train.py --phase c2f --config config/latent_generation.yaml \
  rl.c2f_finetune.num_samples=100 rl.c2f_finetune.epochs=1
```

SLURM: `scripts/slurm_07_rl.sh`. Set `PHASE=sft|c2f|both|joint` before submitting.

---

### Step 8 — Posterior-Collapse Diagnostic

Generates $z$ sequences from the trained $p_\theta$ in causal mode (left-to-right from BOS) and reports whether the latents have collapsed to a handful of repeated tokens — the classic VAE failure mode ([Bowman et al., 2015](https://arxiv.org/abs/1511.06349)).

```bash
python scripts/08_eval_joint.py --checkpoint checkpoints/rl/joint/c2f/step_100
python scripts/08_eval_joint.py --checkpoint checkpoints/decoder --num-samples 20
```

Output (stdout): total $z$ tokens generated, unique-token count and ratio, top-5 most common $z$ tokens, plus a `COLLAPSED` / `NOT collapsed` classification at the 10 % unique-ratio threshold.

---

### Step 9 — Unified NLL Evaluation

Scores a checkpoint on held-out data under a fixed tokenization, reporting nats/word with a bootstrap 95 % CI. The script aborts on a `vocab_size` mismatch between the model and the repo's space tokenizer so cross-model comparisons cannot be silently corrupted.

- `ar` — exact $-\log p(x)$ via shifted CE on raw text. Documents are right-padded and scored in parallel; pad targets are ignored.
- `c2f` — `C2FForCausalLM` training objective as an IWAE-1 lower bound on $\log p(x)$ using gold $z$ read from the test parquet. Reports per-scale nats/token (`z_4, z_3, z_2, z_1, text`) alongside the aggregate.
- `c2f` + `--sft-ckpt` — switches to the **ELBO upper bound** on $-\log p(x) / T$ (`c2f_bound` path). Uses the SFT model as $q_\phi$ to score $\log q(z \mid x)$ and combines with $\log p_\theta(x, z)$; directly comparable to the AR baseline's exact NLL.
- `diffusion` — Monte-Carlo NELBO for the MDLM baseline. Draws `N` samples of $(t, \text{mask})$ per doc and averages the per-position weighted CE. MC draws are packed along the batch dim via `--mc-batch-size` so effective batch is `--batch-size × --mc-batch-size`.

```bash
# AR baseline on raw 32-word documents:
python scripts/09_eval_nll.py \
  --model-kind ar --ckpt checkpoints/ar_baseline/ \
  --test data/tinystoriesv2_shuffled/tinystoriesv2.test.jsonl \
  --limit 2000

# C2F ELBO lower bound on a c2f-format parquet with gold latents + text:
python scripts/09_eval_nll.py \
  --model-kind c2f --ckpt checkpoints/decoder/ \
  --test data/local_generations/c2f_test.parquet \
  --K 1 --out-jsonl results/decoder.per_doc.jsonl

# C2F ELBO upper bound (directly comparable to AR nats/word):
python scripts/09_eval_nll.py \
  --model-kind c2f --ckpt checkpoints/decoder/ \
  --sft-ckpt checkpoints/sft/checkpoint-1172 \
  --test data/local_generations/c2f_test.parquet

# MDLM MC-NELBO (N samples per doc):
python scripts/09_eval_nll.py \
  --model-kind diffusion --ckpt checkpoints/diffusion_baseline/ \
  --test data/tinystoriesv2_shuffled/tinystoriesv2.test.jsonl \
  --N 128 --mc-batch-size 16
```

Passing the same step-5 parquet to `ar` (via `--test` and a text-only view), `c2f`, and `c2f_bound` gives an apples-to-apples comparison on the shared held-out subset.

---

## Configuration

All defaults are defined in `src/config.py` (Pydantic schema). The YAML file only needs to specify overrides. Derived fields (`word_count_constraints`, `text_word_count`) are computed automatically from `scale_lengths`. Top-level `num_gpus` and `seed` propagate to all sections.

Scripts 04, 05, and 06 work without a config file (using built-in defaults); scripts 00, 01, 03, and 07 require or benefit from `--config`.

Ready-to-run configs live under `config/`:

- `latent_generation.yaml` — the base experiment YAML (defaults + project paths).
- `H100_joint_{block,causal}.yaml` — 1 × H100 80 GB. Block vs causal C2F mask.
- `H200_joint_{block,causal}.yaml` — H200 141 GB variants with larger batch sizes.

| Section | Controls |
|---|---|
| `scale_lengths` | Token counts per scale `[2, 4, 8, 16, 32]`; determines hierarchy and `seq_len = 64` |
| `wandb` | W&B enable/disable, project, entity, group, tags, mode |
| `data_prep` | Step 0: dataset name, chunk count, split sizes (`k_prompt`, `k_rl`, etc.) |
| `dataset` | Data layout: `data_dir`, `dataset_name`, split filenames (`prompt_split`, `rl_split`, etc.) |
| `batch` | Steps 1–2: API model, reasoning effort, prompt names, `run_tag`, output dirs |
| `verification` | `strict_word_count` — whether word counts must match exactly |
| `sft` | Steps 3–4: base model, batch size, LR, epochs, `prompt_data` (step 1 output path) |
| `generation` | Step 5: sampling params (temperature, top_p, top_k, max_tokens) |
| `c2f_training` | Step 6: init source, tokenizer type, mask type (`block` or `causal`), FSDP, full HF Trainer config |
| `ar_training` | Step 6b: architecture + HF Trainer config for the no-latents AR baseline; kept in lockstep with `c2f_training` for fair comparison |
| `diffusion_training` | Step 6c: same arch mirror as `ar_training` plus MDLM knobs (`eps_t`, `noise_eps`, `antithetic`, `n_eval_samples`) |
| `rl.sft_rl` | Step 7A: GRPO rollout group size, KL coefficient, format bonus weight |
| `rl.c2f_finetune` | Step 7B: generation and C2F fine-tuning hyperparams |
| `rl.joint` | Step 7 joint: $p_\theta$ LR/WD, save cadence, mask type, malformed reward |

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
│   └── latent_generation.yaml       # experiment config (overrides defaults in src/config.py)
│
├── prompts/                         # prompt templates for batch API (step 1)
│   ├── system_prompts/              #   system prompt text files
│   ├── user_prompts/                #   user prompt templates with {doc} placeholder
│   └── few_shot_examples/           #   few-shot example JSONL files
│
├── batch_api_requests/              # standalone batch API utilities (cost analysis, monitoring)
│
├── scripts/
│   ├── 00_prepare_data.py              # step 0 — download, preprocess, shuffle, split
│   ├── 01_create_batch_requests.py     # step 1 — format documents into batch API JSONL
│   ├── 02_submit_batch.py              # step 2 — submit, monitor, download batch results
│   ├── 03_verify_outputs.py            # step 3 — verify batch outputs → SFT parquet
│   ├── 04_sft_train.py                 # step 4 — HF Trainer SFT (± FSDP)
│   ├── 05_generate_local.py            # step 5 — generate via vLLM or HF, verify, flatten
│   ├── 06_train_decoder.py             # step 6 — C2F pretraining via HF Trainer ± FSDP
│   ├── 06b_train_ar_baseline.py        # step 6b — AR (no-latent) baseline
│   ├── 06c_train_diffusion_baseline.py # step 6c — MDLM masked-diffusion baseline
│   ├── 07_rl_train.py                  # step 7 — ELBO optimisation (sft / c2f / both / joint)
│   ├── 08_eval_joint.py                # step 8 — posterior-collapse diagnostic
│   ├── 09_eval_nll.py                  # step 9 — unified NLL evaluator (ar / c2f / c2f_bound / diffusion)
│   └── slurm_*.sh                      # SLURM job scripts for steps 0–7 + 6b, 6c
│
├── src/
│   ├── config.py            Pydantic config schema + loader (all defaults defined here)
│   ├── common/              load_env, setup_wandb, shared logger, constants, PROJECT_ROOT
│   ├── batch/               step 1–2 — OpenAI client, request creation, submit/monitor, cost
│   ├── data/                step 0 — dataset registry, preprocessing, Pydantic schemas
│   ├── verification/        step 3 — space-based verifier (z_N format + word counts)
│   ├── sft/                 step 3–4 — SFT parquet creation (dataset.py) + Trainer loop (train.py)
│   ├── generation/          step 5 — vLLM / HF inference, prompt loading, flatten_for_c2f, RL parquet builder
│   ├── c2f_model/           C2F model — C2FConfig, C2FForCausalLM, block-prefix + causal attention
│   │   └── training/        steps 6/6b/6c — C2FDataset, ARDataset, space tokenizer, C2FTrainer, DiffusionTrainer
│   ├── rl/                  step 7 — C2FRewardManager (alternating), JointC2FRewardManager, veRL config, phase orchestration
│   └── eval/                step 9 — ar.py / c2f.py / bound.py / diffusion.py (per-model kind evaluators) + common bootstrap
│
├── tests/
│   ├── test_config.py               config loading, propagation, derived fields
│   ├── test_verification.py         verifier logic (pass/fail, word counts, ordering)
│   ├── test_dataset.py              format detection, SFT / C2F / AR dataset creation
│   ├── test_block_mode_leak.py      block-mode attention cannot leak input_ids[i] into logits[i]
│   ├── test_diffusion.py            MDLM loss / schedule / SUBS parameterisation
│   ├── test_eval_batching.py        AR batched eval matches per-doc reference
│   ├── test_eval_bound.py           c2f_bound evaluator (row order, K=1 ELBO sanity)
│   └── test_reward_common.py        reward-manager helpers (layer parsing, C2F input builder)
│
├── data/
│   ├── tinystoriesv2_shuffled/  step 0 output  (chunks, val, test, prompt, rl splits)
│   ├── prompt_data/             step 1 output  (batch API request JSONL)
│   ├── batch_outputs/           step 2 output  (batch API response JSONL)
│   ├── sft_dataset/             step 3 output  (train.parquet)
│   ├── local_generations/       step 5 output  (c2f_train.parquet)
│   └── rl_dataset/              step 7 data    (sft_rl.parquet, c2f_finetune/)
│
├── checkpoints/
│   ├── sft/                     q_φ SFT checkpoint (step 4)
│   ├── decoder/                 p_θ C2F checkpoint (step 6)
│   └── rl/
│       ├── sft/                 q_φ after Phase A GRPO
│       ├── c2f/                 p_θ after Phase B supervised fine-tuning
│       └── joint/               q_φ + per-worker p_θ snapshots from the joint phase
│
├── pyproject.toml               dependencies + optional groups: sft, generation, c2f
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
| 6b — AR baseline | `train/loss`, `eval/loss` on text-only (matched architecture to step 6) |
| 6c — diffusion baseline | MDLM NELBO (`train/loss`) — lower bound on $-\log p(x)$ per masked position |
| 7A — GRPO ($q_\phi$) | Token-normalised reward ($\log p_\theta$) trending up; format pass rate increasing; KL stable |
| 7B — C2F SFT ($p_\theta$) | `eval/loss` decreasing on freshly generated $q_\phi$ samples |
| 7 joint | Per-step `p_loss`, `reward`, `malformed` fraction (via `reward_extra_info`); rolling $p_\theta$ checkpoints kept at `rl.joint.c2f_keep_last_n` |
| 8 — collapse | Unique/total $z$ token ratio, top-5 most common $z$ tokens (stdout) |
| 9 — NLL | Aggregate nats/word + 95 % CI, per-scale nats/token for C2F, per-doc JSONL export. For `c2f_bound`: mean `joint_nll_p` and mean `nll_q` are also printed. |

Each step adds a run tag (`sft`, `generation`, `c2f-pretrain-{block,causal}`, `ar-baseline`, `diffusion-baseline`, `rl-sft`, `rl-c2f`) so runs are filterable in the W&B dashboard. All scripts call `load_env()` and `setup_wandb()` from `src/common/env.py` before any training-framework imports.
