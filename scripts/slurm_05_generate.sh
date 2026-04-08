#!/bin/bash
#SBATCH --job-name=qwen3_4b_generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/generation/qwen3_4b_generate_%j.out
#SBATCH --error=logs/generation/qwen3_4b_generate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 5: Generate latent outputs from SFT model on chunk files.
# Requires: SFT checkpoint and chunk JSONL files from step 0.
# Config: config/latent_generation.yaml (generation + dataset sections).
#
# Override chunks: CHUNKS="0 1" sbatch scripts/slurm_05_generate.sh
# Override samples: sbatch scripts/slurm_05_generate.sh --num-samples 10000

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/generation"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

CHUNKS_ARG=""
if [ -n "$CHUNKS" ]; then
  CHUNKS_ARG="--chunks $CHUNKS"
else
  CHUNKS_ARG="--chunks 0 1 2 3"
fi

python scripts/05_generate_local.py \
  $CHUNKS_ARG \
  --config config/latent_generation.yaml \
  "$@"
