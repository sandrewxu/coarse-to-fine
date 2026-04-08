#!/bin/bash
#SBATCH --job-name=batch_create
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/batch/batch_create_%j.out
#SBATCH --error=logs/batch/batch_create_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 1: Create OpenAI Batch API request JSONL files.
# Requires: prompt split from step 0 (derived from dataset config).
# No GPU required.
#
# The --docs path defaults to dataset.data_dir/dataset.prompt_split from config.
# Override with: DOCS=path/to/docs.jsonl sbatch scripts/slurm_01_batch_create.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs/batch"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DOCS_ARG=""
if [ -n "$DOCS" ]; then
  DOCS_ARG="--docs $DOCS"
fi

python scripts/01_create_batch_requests.py \
  --config config/latent_generation.yaml \
  $DOCS_ARG \
  "$@"
