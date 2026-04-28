#!/bin/bash
# Throughput sweep v4 — memory-pressure probes.
#
# v3 measurements showed peak GPU mem at ~77 GB / 144 GB on H200 with
# rollout_gpu_memory_utilization=0.55. ~67 GB headroom per GPU. Gen is
# 91% of step time, so more KV cache headroom -> more concurrent vLLM
# seqs -> better gen throughput is the working hypothesis.
#
# This sweep anchors on the v2 winner (batch=2048, micro=64, max_seqs=2048)
# and varies gpu_memory_utilization. One combined probe pairs the highest
# util with bumped max_num_seqs to actually use the bigger KV pool.
#
# 15 RL steps per probe (n=14 after dropping warm-up step 1).

set -eo pipefail
cd "$(dirname "$0")/.."

SWEEP_TAG="sweep_v4_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/rl/joint/${SWEEP_TAG}"
mkdir -p "$SWEEP_DIR"

CKPT_BASE="/tmp/c2f_sweep_ckpts/${SWEEP_TAG}"
mkdir -p "$CKPT_BASE"

echo "==========================================================="
echo "Throughput sweep v4 (memory pressure): $SWEEP_TAG"
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

# v2 winner anchor (re-run for stable comparison against the v4 probes,
# since H200 thermal/clock state may differ from when v3 ran).
run_probe "01_v2_winner" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.5

# Modest bump: 0.55 -> 0.70. v3 already measured 0.55, so this isolates the
# next ~22 GB of KV cache.
run_probe "02_util0.70" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.70

# Aggressive: 0.80. ~115 GB to vLLM, leaving ~29 GB for FSDP/activations.
# Training peak in v3 was ~30 GB, so this is the edge.
run_probe "03_util0.80" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.80

# Near-max: 0.85. Likely OOMs during training; if it stays alive it's the
# theoretical ceiling for KV cache on this layout.
run_probe "04_util0.85" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.85

# Combined: high util + more concurrent seqs. Bigger KV pool only helps
# throughput if vLLM is *allowed* to schedule more concurrent seqs.
run_probe "05_util0.80_seqs4096" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=4096 \
  rl.joint.rollout_gpu_memory_utilization=0.80

cleanup_state
rmdir "$CKPT_BASE" 2>/dev/null || true

echo
echo "==========================================================="
echo "Sweep v4 complete."
echo "Reports: scripts/parse_rl_sweep.py $SWEEP_DIR"
echo "==========================================================="
