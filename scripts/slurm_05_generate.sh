#!/bin/bash
#SBATCH --job-name=qwen3_4b_generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/generation/qwen3_4b_generate_%j.out
#SBATCH --error=logs/generation/qwen3_4b_generate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 5: Generate latent outputs from SFT model via vLLM, verify, and flatten for C2F.
# Requires: SFT checkpoint (config generation.model_path) and prompt dataset.
# Config: config/experiments/latent_generation.yaml (generation section).
# Optional: add --num-samples N or --output-dir DIR after the script name below.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/generation"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python scripts/05_generate_local.py --config config/experiments/latent_generation.yaml
