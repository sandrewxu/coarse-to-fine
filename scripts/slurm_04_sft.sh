#!/bin/bash
#SBATCH --job-name=qwen3_4b_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/finetune/qwen3_4b_sft_%j.out
#SBATCH --error=logs/finetune/qwen3_4b_sft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 4: Supervised Fine-Tuning with veRL.
# Requires: data/sft_dataset/train.parquet from step 3 (03_verify_outputs.py).
# Config: config/latent_generation.yaml (sft section).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/finetune"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python scripts/04_sft_train.py \
  --data data/sft_dataset/train.parquet \
  --config config/latent_generation.yaml
