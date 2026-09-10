#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Autonomous Night Pipeline
# Phase 1: Verify & Sync clean qwen_rl_v2
# Phase 2: Run Main Experiment (Table 1: 24 tasks of test_hard for ours_rl_v2 & ours_sft_v2)
# Phase 3: Run Model Scaling Suite (0.8B, 2B, 9B across Base, SFT, RL)
# Phase 4: Aggregate All Tables & Generate Comprehensive Markdown Artifact
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p runs runs/controller_scaling_analysis runs/checkpoints

LOG_FILE="runs/master_night_runner.log"
STATUS_FILE="runs/night_status.json"

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
log "🌙 MARBLE Night Master Pipeline Started"
log "========================================================================"
update_status "initializing" "0%" "Pipeline initialized, awaiting qwen_rl_v2 completion"

# ------------------------------------------------------------------------------
# Phase 1: Wait for qwen_rl_v2 training & sync
# ------------------------------------------------------------------------------
log ">>> [Phase 1/4] Checking status of clean qwen_rl_v2 training on GPU1..."
update_status "training_rl" "10%" "Waiting for qwen_rl_v2 training to complete on GPU1"

while ssh cis_gpu1.utlab.ltd "ps aux | grep -v grep | grep 'train_controller --mode qwen_rl --traces.*runs/checkpoints/qwen_rl_v2' >/dev/null 2>&1"; do
    log ">>> [Phase 1] qwen_rl_v2 training is still running on GPU1. Sleeping 20s..."
    sleep 20
done

log ">>> [Phase 1] Training process on GPU1 finished! Verifying checkpoint..."
ssh cis_gpu1.utlab.ltd "ls -la /data/home/huangzixuan/workspace/MARBLE/runs/checkpoints/qwen_rl_v2/"

# Sync checkpoint to local
log ">>> [Phase 1] Syncing qwen_rl_v2 checkpoint to local repository..."
mkdir -p runs/checkpoints/qwen_rl_v2
rsync -avz "cis_gpu1.utlab.ltd:/data/home/huangzixuan/workspace/MARBLE/runs/checkpoints/qwen_rl_v2/" "runs/checkpoints/qwen_rl_v2/"

# Reload inference server on GPU1 to load freshly trained weights
log ">>> [Phase 1] Reloading inference server on GPU1 with new qwen_rl_v2 adapter..."
ssh cis_gpu1.utlab.ltd "bash -s" << 'RELOAD_SERVER'
set -euo pipefail
pkill -9 -f serve_qwen_openai.py 2>/dev/null || true
tmux kill-session -t qwen_server 2>/dev/null || true
for i in $(seq 1 10); do
    if ! lsof -i :8000 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
tmux new-session -d -s qwen_server "MARBLE_MODEL_PATH=/data/home/huangzixuan/models/Qwen3.5-4B MARBLE_SERVED_MODEL_NAME=Qwen/Qwen3.5-4B MARBLE_SCALE=4b PORT=8000 /data/home/huangzixuan/miniconda3/envs/sigir/bin/python /data/home/huangzixuan/workspace/MARBLE/scripts/serve_qwen_openai.py >> /data/home/huangzixuan/vllm_serve.log 2>&1"
RELOAD_SERVER

log ">>> [Phase 1] Waiting for server on 127.0.0.1:18000/v1/models to become ready..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:18000/v1/models" | grep -q "qwen_rl_v2"; then
        log ">>> [Phase 1] Inference server is READY with qwen_rl_v2!"
        break
    fi
    sleep 2
done

# Quick inference health check
log ">>> [Phase 1] Verifying qwen_rl_v2 test completion..."
curl -s -X POST "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen_rl_v2",
        "messages": [{"role": "user", "content": "You are a memory governance controller. Return a valid json decision: {\"action\": \"none\", \"confidence\": 1.0}"}],
        "max_tokens": 128,
        "temperature": 0.0
    }' | tee -a "$LOG_FILE"
log ""

# ------------------------------------------------------------------------------
# Phase 2: Run Main Experiment (Table 1 Test Hard 24 Tasks)
# ------------------------------------------------------------------------------
log "========================================================================"
log "🚀 [Phase 2/4] Executing Main Experiment (Table 1 on 24 test_hard tasks)"
log "========================================================================"
update_status "evaluating_main_table1" "25%" "Executing Table 1 evaluation (24 tasks)"

# 1. Evaluate Ours-RL v2 on all 24 tasks
log ">>> [Phase 2.1] Running ours_rl_v2 across 24 test_hard tasks..."
mkdir -p runs/test_hard_ours_rl_v2
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_rl \
    --split test_hard \
    --concurrency 8 \
    --controller-checkpoint runs/checkpoints/qwen_rl_v2 \
    --qwen-api-base "http://127.0.0.1:18000/v1" \
    --qwen-api-model qwen_rl_v2 \
    --out runs/test_hard_ours_rl_v2 \
    2>&1 | tee -a "$LOG_FILE"

# 2. Evaluate Ours-SFT v2 on all 24 tasks
log ">>> [Phase 2.2] Running ours_sft_v2 across 24 test_hard tasks..."
mkdir -p runs/test_hard_ours_sft_v2
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_sft \
    --split test_hard \
    --concurrency 8 \
    --controller-checkpoint runs/checkpoints/qwen_sft_v2 \
    --qwen-api-base "http://127.0.0.1:18000/v1" \
    --qwen-api-model qwen_sft_v2 \
    --out runs/test_hard_ours_sft_v2 \
    2>&1 | tee -a "$LOG_FILE"

# 3. Aggregate Table 1 Master Results
log ">>> [Phase 2.3] Aggregating Table 1 Master Benchmark Results..."
./.venv/bin/python scripts/aggregate_hard_master_table.py \
    --run-dir runs/test_hard_ours_rl_v2 runs/test_hard_ours_sft_v2 runs/test_hard_ours_sft runs/test_hard_ours_base runs/test_hard_baselines \
    --manifest configs/experiments/multiagentbench_hard_frozen.json \
    --split test_hard \
    --out-json runs/table1_master_results.json \
    2>&1 | tee -a "$LOG_FILE"

update_status "evaluating_main_table1_done" "50%" "Main Table 1 completed successfully"

# ------------------------------------------------------------------------------
# Phase 3: Parameter Scaling Ablation Suite (0.8B, 2B, 9B)
# ------------------------------------------------------------------------------
log "========================================================================"
log "🚀 [Phase 3/4] Running Parameter Scaling Ablation Suite (0.8B, 2B, 9B)"
log "========================================================================"
update_status "scaling_ablation" "55%" "Running parameter scaling ablation suite"

./scripts/run_controller_scaling_analysis.sh \
    --scales "0.8b 2b 9b" \
    --methods "ours_base ours_sft ours_rl" \
    --concurrency 6 \
    --seed 42 \
    --out runs/controller_scaling_analysis \
    2>&1 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# Phase 4: Final Aggregation & Comprehensive Summary
# ------------------------------------------------------------------------------
log "========================================================================"
log "📊 [Phase 4/4] Final Aggregation & Summary Generation"
log "========================================================================"
update_status "aggregating_all" "90%" "Aggregating all experimental results"

./.venv/bin/python scripts/aggregate_scaling_analysis_table.py \
    --base-dir runs/controller_scaling_analysis \
    --scales 0.8b 2b 9b \
    --methods ours_base ours_sft ours_rl \
    --json-out runs/controller_scaling_analysis/scaling_analysis_report.json \
    2>&1 | tee -a "$LOG_FILE"

update_status "completed" "100%" "All experiments completed successfully!"
log "========================================================================"
log "🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY AT $(date '+%Y-%m-%d %H:%M:%S')"
log "========================================================================"
