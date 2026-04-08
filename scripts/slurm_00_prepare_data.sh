#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/data_prep/prepare_data_%j.out
#SBATCH --error=logs/data_prep/prepare_data_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 0: Download, preprocess, shuffle, and split a HuggingFace dataset.
# Produces sharded JSONL chunks + val/test/prompt/rl splits.
# No GPU required.
#
# Override dataset: DATASET=fineweb_edu_10bt sbatch scripts/slurm_00_prepare_data.sh
# Override memory:  MEMORY=32 sbatch scripts/slurm_00_prepare_data.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/data_prep"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATASET=${DATASET:-tinystoriesv2}
MEMORY=${MEMORY:-8}

python scripts/00_prepare_data.py \
  --dataset "$DATASET" \
  --memory "$MEMORY" \
  --config config/latent_generation.yaml \
  "$@"
