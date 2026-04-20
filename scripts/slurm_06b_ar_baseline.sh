#!/bin/bash
#SBATCH --job-name=ar_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/pretrain/ar_baseline_%j.out
#SBATCH --error=logs/pretrain/ar_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 6b: No-latents AR baseline (matched to step 6 C2F pretraining).
# Reads the same parquet, uses the same space tokenizer, takes the same
# 95/5 split (seed=42). Directly comparable via scripts/09_eval_nll.py.
#
# For multi-GPU, change --gpus and NUM_GPUS below:
#   #SBATCH --gpus=h100:4
#   NUM_GPUS=4

set -e
# Under sbatch, SLURM copies this script to /var/spool/slurmd/... so
# ${BASH_SOURCE[0]} no longer points to the repo. SLURM_SUBMIT_DIR is the
# directory `sbatch` was invoked from — use it when available.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
mkdir -p "$PROJECT_ROOT/logs/pretrain"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

NUM_GPUS=${NUM_GPUS:-1}
CONFIG=${CONFIG:-config/H100_joint_causal.yaml}
DATA=${DATA:-data/local_generations/c2f_train.parquet}

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch --num_processes="$NUM_GPUS" \
    scripts/06b_train_ar_baseline.py \
    --data "$DATA" \
    --config "$CONFIG"
else
  python scripts/06b_train_ar_baseline.py \
    --data "$DATA" \
    --config "$CONFIG"
fi
