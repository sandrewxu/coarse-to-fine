#!/bin/bash
#SBATCH --job-name=rl_joint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/rl/joint/slurm_%j.out
#SBATCH --error=logs/rl/joint/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Step 7: ELBO Optimisation — default PHASE=joint (REINFORCE++ on q_φ + MLE on p_θ).
#
# Also works for the other phases (set PHASE= at the top of the env):
#   PHASE=joint — Simultaneous REINFORCE++ on q_φ + MLE on p_θ (default)
#   PHASE=sft   — Phase A only (GRPO on q_φ, p_θ frozen)
#   PHASE=c2f   — Phase B only (supervised fine-tuning of p_θ, q_φ frozen)
#   PHASE=both  — Phase A then Phase B
#
# Requires:
#   - data/tinystoriesv2_shuffled/tinystoriesv2.rl.jsonl   (RL training docs, from step 0)
#   - data/tinystoriesv2_shuffled/tinystoriesv2.val.jsonl  (held-out val docs,  from step 0)
#   - checkpoints/sft/...                                  (q_φ starting point, from step 4)
#   - checkpoints/decoder/                                 (p_θ starting point, from step 6)
#
# Config: config/latent_generation.yaml (rl section).
#
# Logging:
#   Output is tee'd to logs/rl/joint/rl_joint_<timestamp>.log so it survives
#   interactive-session disconnects. Under SLURM, the same content also lands
#   in logs/rl/joint/rl_joint_<jobid>.{out,err}.
#
#   DEBUG=1   Enable verbose logging (LOG_LEVEL=DEBUG, VERL_LOGGING_LEVEL=INFO).
#             Default is INFO / WARN (veRL upstream default).

set -eo pipefail
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

PHASE=${PHASE:-joint}
CONFIG=${CONFIG:-config/H200_joint_causal.yaml}

# Log directory is phase-aware so the joint runs don't intermix with sft/c2f.
LOG_DIR="logs/rl/${PHASE}"
mkdir -p "$LOG_DIR"

# Tee all subsequent output to a per-run log file. Under SLURM, --output still
# captures this stream via the kernel pipe, so we get both slurm .out and the
# timestamped log; outside SLURM the tee is the only capture.
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="$LOG_DIR/rl_${PHASE}_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[slurm_07_rl.sh] logging to $LOG_FILE"


# Debug verbosity toggle.
if [ "${DEBUG:-0}" = "1" ]; then
  export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
  export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
  echo "[slurm_07_rl.sh] DEBUG mode: LOG_LEVEL=$LOG_LEVEL VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL"
fi

# Reward managers locate the experiment YAML via this env var (Ray workers
# inherit a stripped env and can't reliably see the caller's CWD).
export C2F_CONFIG_PATH="$PROJECT_ROOT/$CONFIG"

echo "[slurm_07_rl.sh] phase=$PHASE config=$CONFIG"
uv run --no-sync python scripts/07_rl_train.py \
  --phase "$PHASE" \
  --config "$CONFIG" \
  "$@"
