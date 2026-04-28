#!/bin/bash
# Throughput sweep v3 — push MFU past v2's 24%.
#
# v2 winner: batch=2048, micro=64, max_seqs=2048, mem=0.5 → 4557 tok/s, 24% MFU.
# Bottleneck is gen (91% of step time). Hypotheses:
#   - bigger train_batch lets vLLM pack KV slots more efficiently
#   - higher max_num_batched_tokens lets prefill chunks be wider
#   - alt parallelism (TP=2, DP=4) might amortize gen cost differently on a 4B
#
# 15 RL steps per probe (n=14 measurements after dropping warm-up step 1).
# Each probe gets a unique checkpoint dir + resume_mode=disable for isolation.

set -eo pipefail
cd "$(dirname "$0")/.."

SWEEP_TAG="sweep_v3_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/rl/joint/${SWEEP_TAG}"
mkdir -p "$SWEEP_DIR"

CKPT_BASE="/tmp/c2f_sweep_ckpts/${SWEEP_TAG}"
mkdir -p "$CKPT_BASE"

echo "==========================================================="
echo "Throughput sweep v3: $SWEEP_TAG"
echo "Sweep dir: $SWEEP_DIR"
echo "Per-probe ckpt base: $CKPT_BASE"
echo "==========================================================="

COMMON=(
  "++trainer.total_training_steps=15"
  "++trainer.test_freq=99999"
  "++trainer.save_freq=99999"
  "++trainer.val_before_train=False"
  "rl.joint.resume_mode=disable"
)

cleanup_state () {
  pkill -f "main_ppo|07_rl_train|raylet|ray::|gcs_server" 2>/dev/null || true
  rm -f /tmp/verl_vllm_zmq_*.ipc
  sleep 5
}

run_probe () {
  local name="$1"; shift
  local logfile="$SWEEP_DIR/${name}.log"
  local ckpt="$CKPT_BASE/${name}"

  echo
  echo "----- [$(date '+%H:%M:%S')] probe: $name -----"
  echo "log:  $logfile"

  cleanup_state

  SLURM_JOB_ID="${SWEEP_TAG}_${name}" \
      bash scripts/slurm_07_rl.sh \
          "rl.joint.checkpoint_dir=$ckpt" \
          "rl.joint.c2f_save_dir=$ckpt/c2f" \
          "$@" "${COMMON[@]}" \
      > "$logfile" 2>&1 \
      && rc=$? || rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "[OK]   $name"
  else
    echo "[FAIL] $name (rc=$rc) — see $logfile"
  fi

  rm -rf "$ckpt"
}

# -------------------------------------------------------------------- #
# v3 sweep: targeted at gen throughput / MFU                           #
# -------------------------------------------------------------------- #

# v2 winner re-run with 15 steps for stable timing (and as the comparison anchor).
run_probe "01_v2_winner" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.5

# Push batch further: 4096. Was 24% MFU at 2048; expect higher.
run_probe "02_batch4096" \
  rl.joint.train_batch_size=4096 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=4096 \
  rl.joint.rollout_gpu_memory_utilization=0.55

# Even bigger batch — does it OOM, or keep scaling?
run_probe "03_batch8192" \
  rl.joint.train_batch_size=8192 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=8192 \
  rl.joint.rollout_gpu_memory_utilization=0.6

# Wider prefill chunks (default 32k) at v2 winner config.
# Should help if prefill is bottleneck; should be neutral otherwise.
run_probe "04_tokens128k" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.5 \
  rl.joint.rollout_max_num_batched_tokens=131072

# Alternative parallelism: TP=2 / DP=4. Verl bug patch should handle DP=4.
# vLLM denylist patch should handle the executor backend.
run_probe "05_TP2_DP4" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.5 \
  rl.joint.rollout_tensor_parallel_size=2 \
  rl.joint.rollout_data_parallel_size=4

cleanup_state
rmdir "$CKPT_BASE" 2>/dev/null || true

echo
echo "==========================================================="
echo "Sweep v3 complete."
echo "Reports: scripts/parse_rl_sweep.py $SWEEP_DIR"
echo "==========================================================="
