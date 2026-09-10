#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE GPU1 Master Autonomous Pipeline (16-Worker Concurrency)
# Complete Native Rootless Stack on cis_gpu1.utlab.ltd (NVIDIA H20 96GB + 188GB RAM)
# 
# Stages:
# 1. Verification & Rollout 4 (16 parallel workers)
# 2. Calibrated Hybrid GRPO Training (runs/checkpoints/qwen_rl_v3)
# 3. Full 24-Task Test-Hard Evaluation: Ours-RL-v3 & Ours-SFT-v3 (16 workers)
# 4. Ablations & Parameter Sensitivity Analysis (16 workers)
# 5. Table 1, Table 2 & Master Results Aggregation
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PATH="/data/home/huangzixuan/miniconda3/envs/sigir/bin:/data/home/huangzixuan/miniconda3/envs/pg_env/bin:$PATH"
export PYTHONPATH="$REPO_DIR"
export PYTHON_BIN="/data/home/huangzixuan/miniconda3/envs/sigir/bin/python"
export MARBLE_DB_RUNTIME="native"

mkdir -p runs runs/checkpoints runs/rl_v3_rollouts

LOG_FILE="runs/gpu1_master_pipeline.log"
STATUS_FILE="runs/gpu1_status.json"

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
log "🚀 MARBLE GPU1 Master Pipeline Started (16-Worker Concurrency, H20 96GB)"
log "========================================================================"

# Pre-flight cleanup
./scripts/native_db_ctl.sh cleanup_all

# Ensure vLLM is running on port 8000
log ">>> Verifying vLLM inference server on 127.0.0.1:8000..."
./scripts/gpu1_vllm_ctl.sh start

# ------------------------------------------------------------------------------
# Stage 1: Ensure Rollouts 1~4 are complete (Unified Dynamic Multi-Rollout Queue)
# ------------------------------------------------------------------------------
OUT_BASE="runs/rl_v3_rollouts"
MANIFEST="configs/experiments/rl_train_12tasks_hard.json"

mkdir -p "$OUT_BASE/rollout_1" "$OUT_BASE/rollout_2" "$OUT_BASE/rollout_3" "$OUT_BASE/rollout_4"

log ">>> [Stage 1/5] Running Global Dynamic Queue across Rollouts (16-Worker Concurrency)..."
update_status "rollout_multi_queue" "25%" "Executing dynamic cross-rollout queue (16 workers)"

./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_sft \
    --manifest "$MANIFEST" \
    --split train_hard \
    --concurrency 16 \
    --seed 42 \
    --qwen-temperature 0.7 \
    --controller-checkpoint runs/checkpoints/qwen_sft_v3 \
    --qwen-api-model qwen_sft_v3 \
    --multi-out "$OUT_BASE/rollout_3,$OUT_BASE/rollout_4"

log ">>> Rollouts 1~4 verified complete via Global Dynamic Queue!"

# ------------------------------------------------------------------------------
# Stage 2: Train qwen_rl_v3 with calibrated hybrid GRPO on H20
# ------------------------------------------------------------------------------
log ">>> [Stage 2/5] Stopping vLLM to free 96GB VRAM for GRPO training..."
update_status "grpo_training" "45%" "Training qwen_rl_v3 with calibrated hybrid GRPO on H20"
./scripts/gpu1_vllm_ctl.sh stop
kill $(lsof -t -i:8000 2>/dev/null || true) 2>/dev/null || true

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
done < <(find "$OUT_BASE" -name "summary.json" | sort)

log ">>> Found ${#TRACES[@]} trajectories for GRPO v3 update."

$PYTHON_BIN -m marble.experiments.train_controller \
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

log "✅ Qwen RL v3 Training Completed! Checkpoint saved at $OUT_CKPT."

# ------------------------------------------------------------------------------
# Stage 3: Master Hard Evaluation on 24 test_hard Tasks (Concurrency 16)
# ------------------------------------------------------------------------------
log ">>> [Stage 3/5] Restarting vLLM inference server with both SFT and RL adapters..."
update_status "evaluation_rl" "60%" "Evaluating Ours-RL-v3 on 24 test_hard tasks (16 workers)"
./scripts/gpu1_vllm_ctl.sh start

MASTER_MANIFEST="configs/experiments/multiagentbench_hard_frozen.json"

log ">>> Evaluating Ours-RL-v3 on 24 test_hard tasks at 16 concurrency..."
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_rl \
    --manifest "$MASTER_MANIFEST" \
    --split test_hard \
    --concurrency 16 \
    --controller-checkpoint runs/checkpoints/qwen_rl_v3 \
    --qwen-api-model qwen_rl_v3 \
    --out runs/test_hard_ours_rl_v3

log ">>> Evaluating Ours-SFT-v3 on 24 test_hard tasks at 16 concurrency..."
update_status "evaluation_sft" "75%" "Evaluating Ours-SFT-v3 on 24 test_hard tasks (16 workers)"
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_sft \
    --manifest "$MASTER_MANIFEST" \
    --split test_hard \
    --concurrency 16 \
    --controller-checkpoint runs/checkpoints/qwen_sft_v3 \
    --qwen-api-model qwen_sft_v3 \
    --out runs/test_hard_ours_sft_v3

# ------------------------------------------------------------------------------
# Stage 4: Ablations & Parameter Sensitivity Sweeps (Concurrency 16)
# ------------------------------------------------------------------------------
log ">>> [Stage 4/5] Running Visibility Ablation (Private to Global) at 16 concurrency..."
update_status "ablations" "85%" "Running ablations and sensitivity analyses (16 workers)"
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_private_to_global \
    --ablation "visibility:private_to_global" \
    --manifest "$MASTER_MANIFEST" \
    --split test_hard \
    --concurrency 16 \
    --controller-checkpoint runs/checkpoints/qwen_rl_v3 \
    --qwen-api-model qwen_rl_v3 \
    --out runs/ablation_private_to_global

# ------------------------------------------------------------------------------
# Stage 5: Master Table Aggregation
# ------------------------------------------------------------------------------
log ">>> [Stage 5/5] Aggregating Table 1 & Master Reports..."
update_status "aggregating" "95%" "Generating Table 1 & master evaluation report"

$PYTHON_BIN scripts/aggregate_hard_master_table.py \
    --run-dir runs \
    --out-json runs/table1_v3_results.json

update_status "completed" "100%" "All v3 experiments and evaluations completed on GPU1"
log "========================================================================"
log "🎉 MARBLE GPU1 MASTER PIPELINE FULLY COMPLETED!"
log "========================================================================"
