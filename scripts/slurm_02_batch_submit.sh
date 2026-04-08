#!/bin/bash
#SBATCH --job-name=batch_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/batch/batch_submit_%j.out
#SBATCH --error=logs/batch/batch_submit_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 2: Submit batch request to OpenAI and monitor until complete.
# Requires: request JSONL from step 1.
# No GPU required. Long timeout to allow batch API polling.
#
# Set INPUT_FILE to the sft.jsonl produced by step 1, e.g.:
#   INPUT_FILE=data/prompt_data/gpt-5-nano-2025-08-07/docs_0000010000/gemini-3-pro-7/latent-generation/sft.jsonl

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/batch"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

if [ -z "$INPUT_FILE" ]; then
  echo "Error: INPUT_FILE not set. Set it to the sft.jsonl from step 1." >&2
  exit 1
fi

python scripts/02_submit_batch.py \
  --input "$INPUT_FILE" \
  --config config/latent_generation.yaml \
  --monitor \
  "$@"
