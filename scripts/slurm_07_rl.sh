#!/bin/bash
#SBATCH --job-name=rl_elbo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/rl/rl_elbo_%j.out
#SBATCH --error=logs/rl/rl_elbo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 7: ELBO Optimisation — GRPO on q_φ (SFT) / Supervised on p_θ (C2F).
#
# Requires:
#   - data/sft_dataset/train.parquet         (from step 3)
#   - checkpoints/sft/sft_backup_3epoch/...  (Phase A: q_φ start)
#   - checkpoints/decoder/                   (Phase A: frozen p_θ reward)
#   - checkpoints/rl/sft/                    (Phase B: updated q_φ, after Phase A)
#
# Config: config/experiments/latent_generation.yaml (rl section).
#
# Phase selection (default: both):
#   PHASE=sft   — only Phase A (GRPO on q_φ)
#   PHASE=c2f   — only Phase B (supervised fine-tuning of p_θ)
#   PHASE=both  — Phase A then Phase B

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/rl"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

PHASE=${PHASE:-both}
CONFIG=${CONFIG:-config/experiments/latent_generation.yaml}

# Export config path so the reward manager can locate the experiment YAML
# without needing to hard-code the path inside the Python module.
export C2F_CONFIG_PATH="$PROJECT_ROOT/$CONFIG"

echo "Running RL step 7: phase=$PHASE, config=$CONFIG"
python scripts/07_rl_train.py \
  --phase "$PHASE" \
  --config "$CONFIG" \
  "$@"
