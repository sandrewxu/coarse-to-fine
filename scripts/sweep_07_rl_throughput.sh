#!/bin/bash
# Throughput sweep v2 for step 7 joint RL (REINFORCE on q_φ + MLE on p_θ).
#
# Each named probe runs N RL steps with validation/saves disabled, isolated
# from siblings (unique checkpoint_dir + resume_mode=disable). Per-probe log
# under logs/rl/joint/sweep_TS/<name>.log.
#
# Inherits config/H200_joint_causal.yaml as the base. Per-probe overrides
# (rl.joint.* and ++trainer.*) are layered on via the slurm script's
# pass-through to scripts/07_rl_train.py overrides.
#
# Reads:
#   - patches/* must already be applied (verl DP fix + vllm executor fix)
#   - vllm_async_server.py:198 max_model_len clobber must be made conditional
#     (otherwise rollout_max_num_seqs has limited effect)
#
# Run: bash scripts/sweep_07_rl_throughput.sh
# Parse: scripts/parse_rl_sweep.py logs/rl/joint/sweep_<TS>/

set -eo pipefail
cd "$(dirname "$0")/.."

SWEEP_TAG="sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/rl/joint/${SWEEP_TAG}"
mkdir -p "$SWEEP_DIR"

# Per-probe checkpoint dirs land under here so each probe is independent.
# We `rm -rf` each one after the probe finishes to keep disk usage bounded
# (verl saves on final step regardless of save_freq).
CKPT_BASE="/tmp/c2f_sweep_ckpts/${SWEEP_TAG}"
mkdir -p "$CKPT_BASE"

echo "==========================================================="
echo "Throughput sweep v2: $SWEEP_TAG"
echo "Sweep dir: $SWEEP_DIR"
echo "Per-probe ckpt base: $CKPT_BASE"
echo "==========================================================="

# Common overrides applied to every probe.
COMMON=(
  "++trainer.total_training_steps=10"
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
  echo "ckpt: $ckpt"

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

  # Reclaim disk: each probe saves a final checkpoint (~8 GB actor + optim).
  rm -rf "$ckpt"
}

# -------------------------------------------------------------------- #
# v2 sweep matrix                                                      #
# All probes use train_batch=512 baseline unless varied. Each probe is #
# fully independent (resume_mode=disable + unique checkpoint_dir).     #
# -------------------------------------------------------------------- #

# Re-baseline: previous winner (06_micro64) with the new fixes applied.
run_probe "01_winner_recheck" \
  rl.joint.train_batch_size=512 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=512 \
  rl.joint.rollout_gpu_memory_utilization=0.5

# Push max_num_seqs (now meaningful with max_model_len=512 fix).
run_probe "02_seqs1024" \
  rl.joint.train_batch_size=512 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=1024 \
  rl.joint.rollout_gpu_memory_utilization=0.5

run_probe "03_seqs2048" \
  rl.joint.train_batch_size=512 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.6

# Bigger train_batch — denoise REINFORCE; needs proportionally bigger seqs.
run_probe "04_batch1024" \
  rl.joint.train_batch_size=1024 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=1024 \
  rl.joint.rollout_gpu_memory_utilization=0.5

run_probe "05_batch2048" \
  rl.joint.train_batch_size=2048 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=2048 \
  rl.joint.rollout_gpu_memory_utilization=0.5

# Bigger PPO micro — fewer FSDP all-gathers per step.
run_probe "06_micro128" \
  rl.joint.train_batch_size=1024 \
  rl.joint.ppo_micro_batch_size_per_gpu=128 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=1024 \
  rl.joint.rollout_gpu_memory_utilization=0.5

# vLLM CUDA graphs — biggest single potential win for gen if it works.
# Setting enforce_eager=False is risky with verl's external executor;
# this probe will tell us empirically if it works in our setup.
run_probe "07_no_eager" \
  rl.joint.train_batch_size=512 \
  rl.joint.ppo_micro_batch_size_per_gpu=64 \
  rl.joint.c2f_micro_batch_size=256 \
  rl.joint.rollout_max_num_seqs=512 \
  rl.joint.rollout_gpu_memory_utilization=0.5 \
  ++actor_rollout_ref.rollout.enforce_eager=False

cleanup_state
rmdir "$CKPT_BASE" 2>/dev/null || true

echo
echo "==========================================================="
echo "Sweep complete."
echo "Reports: scripts/parse_rl_sweep.py $SWEEP_DIR"
echo "==========================================================="
