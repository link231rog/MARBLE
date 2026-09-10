#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Autonomous v3 Pipeline
# 1. Verify & Sync clean qwen_sft_v3
# 2. Rollout 12 Hard Tasks x 4 Waves (48 trajectories) using qwen_sft_v3
# 3. Train qwen_rl_v3 with calibrated hybrid GRPO on GPU1
# 4. Master Evaluation on 24 test_hard tasks: Ours-SFT-v3 vs Ours-RL-v3
# 5. Aggregate Table 1 Master Report
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p runs runs/checkpoints runs/rl_v3_rollouts

LOG_FILE="runs/master_v3_runner.log"
STATUS_FILE="runs/v3_status.json"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

update_status() {
    local stage="$1"
    local progress="$2"
    local details="$3"
    cat << STAT_EOF > "$STATUS_FILE"
{
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
  "stage": "$stage",
  "progress": "$progress",
  "details": "$details"
}
STAT_EOF
}

log "========================================================================"
log "🚀 MARBLE v3 Master Pipeline Started"
log "========================================================================"
update_status "sft_training" "15%" "Waiting for qwen_sft_v3 training to finish on GPU1"

# ------------------------------------------------------------------------------
# Phase 1: Wait for qwen_sft_v3 training & sync
# ------------------------------------------------------------------------------
log ">>> [Phase 1/5] Checking status of qwen_sft_v3 training on GPU1..."
while ssh cis_gpu1.utlab.ltd "ps aux | grep -v grep | grep 'train_controller.*qwen_sft.*runs/checkpoints/qwen_sft_v3' >/dev/null 2>&1"; do
    log ">>> [Phase 1] qwen_sft_v3 training is still running on GPU1. Sleeping 25s..."
    sleep 25
done

log ">>> [Phase 1] SFT training finished! Verifying checkpoint on GPU1..."
ssh cis_gpu1.utlab.ltd "ls -la /data/home/huangzixuan/workspace/MARBLE/runs/checkpoints/qwen_sft_v3/"

log ">>> [Phase 1] Syncing qwen_sft_v3 checkpoint to local repository..."
mkdir -p runs/checkpoints/qwen_sft_v3
rsync -avz "cis_gpu1.utlab.ltd:/data/home/huangzixuan/workspace/MARBLE/runs/checkpoints/qwen_sft_v3/" "runs/checkpoints/qwen_sft_v3/"

log ">>> [Phase 1] Reloading inference server on GPU1 with new qwen_sft_v3 adapter..."
./scripts/gpu1_vllm_ctl.sh start

# ------------------------------------------------------------------------------
# Phase 2: Rollout 12 tasks x 4 Waves for RL Training
# ------------------------------------------------------------------------------
log ">>> [Phase 2/5] Starting 4-wave rollouts across 12 train_hard tasks..."
update_status "rollout" "30%" "Executing 4 rollouts across 12 hard training tasks"

MANIFEST="configs/experiments/rl_train_12tasks_hard.json"
OUT_BASE="runs/rl_v3_rollouts"

ROLLOUTS=(1 2 3 4)
for ROLLOUT in "${ROLLOUTS[@]}"; do
    ROLLOUT_OUT="$OUT_BASE/rollout_${ROLLOUT}"
    log ">>> [Phase 2] Dispatching Rollout $ROLLOUT/4 (12 tasks, concurrency 6) to $ROLLOUT_OUT..."
    TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
        --baseline ours_sft \
        --manifest "$MANIFEST" \
        --split train_hard \
        --concurrency 6 \
        --seed 42 \
        --qwen-temperature 0.7 \
        --controller-checkpoint runs/checkpoints/qwen_sft_v3 \
        --qwen-api-model qwen_sft_v3 \
        --out "$ROLLOUT_OUT"
    log ">>> [Phase 2] Rollout $ROLLOUT/4 completed!"
done

# ------------------------------------------------------------------------------
# Phase 3: Train qwen_rl_v3 with calibrated hybrid GRPO on GPU1
# ------------------------------------------------------------------------------
log ">>> [Phase 3/5] Stopping vLLM to free 100% VRAM for GRPO..."
update_status "grpo_training" "55%" "Training qwen_rl_v3 with calibrated hybrid GRPO on GPU1"
./scripts/gpu1_vllm_ctl.sh stop
kill $(lsof -t -i:18000) 2>/dev/null || true

log ">>> [Phase 3] Syncing rollout traces to cis_gpu1.utlab.ltd..."
ssh cis_gpu1.utlab.ltd "mkdir -p /data/home/huangzixuan/workspace/MARBLE/$OUT_BASE"
rsync -avz "$OUT_BASE/" "cis_gpu1.utlab.ltd:/data/home/huangzixuan/workspace/MARBLE/$OUT_BASE/"

log ">>> [Phase 3] Launching calibrated GRPO training on GPU1..."
ssh cis_gpu1.utlab.ltd bash -s "$OUT_BASE" << 'REMOTE_GRPO_V3'
set -euo pipefail
REMOTE_OUT_BASE="$1"
cd /data/home/huangzixuan/workspace/MARBLE
source /data/home/huangzixuan/miniconda3/bin/activate sigir 2>/dev/null || true
export PYTHONPATH=.

OUT_CKPT="runs/checkpoints/qwen_rl_v3"
mkdir -p "$OUT_CKPT"

TRACES=()
REWARDS=()
while IFS= read -r summary_file; do
    trace_file="${summary_file%summary.json}memory_trace.jsonl"
    if [ -f "$trace_file" ]; then
        TRACES+=("$trace_file")
        REWARDS+=("$summary_file")
    fi
done < <(find "$REMOTE_OUT_BASE" -name "summary.json" | sort)

echo "Found ${#TRACES[@]} traces for GRPO v3 update."

python -m marble.experiments.train_controller \
    --mode qwen_rl \
    --traces "${TRACES[@]}" \
    --rewards "${REWARDS[@]}" \
    --init runs/checkpoints/qwen_sft_v3 \
    --base-model /data/home/huangzixuan/models/Qwen3.5-4B \
    --epochs 1 \
    --max-len 4096 \
    --advantage-mode hybrid \
    --beta 0.75 \
    --lambda-cost 0.005 \
    --target-bonus 0.50 \
    --target-card-budget 20 \
    --gamma-density 0.02 \
    --harmful-penalty 0.05 \
    --out "$OUT_CKPT"

echo "✅ Qwen RL v3 Training Completed!"
REMOTE_GRPO_V3

log ">>> [Phase 3] Syncing qwen_rl_v3 checkpoint back to local..."
mkdir -p runs/checkpoints/qwen_rl_v3
rsync -avz "cis_gpu1.utlab.ltd:/data/home/huangzixuan/workspace/MARBLE/runs/checkpoints/qwen_rl_v3/" "runs/checkpoints/qwen_rl_v3/"

# ------------------------------------------------------------------------------
# Phase 4: Master Hard Evaluation (24 test_hard tasks for SFT-v3 & RL-v3)
# ------------------------------------------------------------------------------
log ">>> [Phase 4/5] Starting inference server on GPU1 with both adapters..."
update_status "evaluation" "70%" "Evaluating Ours-RL-v3 and Ours-SFT-v3 on 24 test_hard tasks"
./scripts/gpu1_vllm_ctl.sh start

MASTER_MANIFEST="configs/experiments/multiagentbench_hard_frozen.json"

log ">>> [Phase 4] Evaluating Ours-RL-v3 on 24 test_hard tasks..."
TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_rl \
    --manifest "$MASTER_MANIFEST" \
    --split test_hard \
    --concurrency 6 \
    --controller-checkpoint runs/checkpoints/qwen_rl_v3 \
    --qwen-api-model qwen_rl_v3 \
    --out runs/test_hard_ours_rl_v3

log ">>> [Phase 4] Evaluating Ours-SFT-v3 on 24 test_hard tasks..."
TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_sft \
    --manifest "$MASTER_MANIFEST" \
    --split test_hard \
    --concurrency 6 \
    --controller-checkpoint runs/checkpoints/qwen_sft_v3 \
    --qwen-api-model qwen_sft_v3 \
    --out runs/test_hard_ours_sft_v3

# ------------------------------------------------------------------------------
# Phase 5: Master Table Aggregation
# ------------------------------------------------------------------------------
log ">>> [Phase 5/5] Aggregating Table 1 Master Results..."
update_status "completed" "100%" "All v3 experiments and evaluations completed"

./.venv/bin/python scripts/aggregate_hard_master_table.py \
    --runs-dir runs \
    --out runs/table1_v3_results.json

log "========================================================================"
log "🎉 MARBLE v3 MASTER PIPELINE FULLY COMPLETED!"
log "========================================================================"
