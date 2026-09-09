#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MANIFEST="configs/experiments/rl_smoke_train_6tasks.json"
OUT_BASE="${OUT_BASE:-runs/grpo_recovery_smoke_20260909}"
CKPT="${CONTROLLER_CHECKPOINT:-runs/checkpoints/qwen_sft_v2}"
MODEL_NAME="${QWEN_API_MODEL:-qwen_sft_v2}"
mkdir -p "$OUT_BASE"

echo "========================================================================"
echo "🚀 RL Smoke Verification: 6 Tasks x 4 Rollouts (Fixed Seed 42, Temp 0.7)"
echo "  Manifest:    $MANIFEST"
echo "  Checkpoint:  $CKPT"
echo "  Model:       $MODEL_NAME"
echo "  Output Base: $OUT_BASE"
echo "========================================================================"

trap './scripts/gpu1_vllm_ctl.sh stop >/dev/null 2>&1 || true; kill $(lsof -t -i:18000) >/dev/null 2>&1 || true' EXIT INT TERM

echo ">>> Ensuring minimal VRAM Qwen Controller is active on GPU1..."
./scripts/gpu1_vllm_ctl.sh start

ROLLOUTS=(1 2 3 4)
for ROLLOUT in "${ROLLOUTS[@]}"; do
    ROLLOUT_OUT="$OUT_BASE/rollout_${ROLLOUT}"
    echo ""
    echo ">>> [Rollout $ROLLOUT/4] Dispatching 6 tasks (concurrency 6) to $ROLLOUT_OUT..."
    TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
        --baseline ours_sft \
        --manifest "$MANIFEST" \
        --split train_hard \
        --concurrency 6 \
        --seed 42 \
        --qwen-temperature 0.7 \
        --controller-checkpoint "$CKPT" \
        --qwen-api-model "$MODEL_NAME" \
        --out "$ROLLOUT_OUT"
    echo ">>> [Rollout $ROLLOUT/4] Finished."
done

echo ""
echo ">>> Stopping Qwen Controller on GPU1 to free 100% VRAM for GRPO..."
./scripts/gpu1_vllm_ctl.sh stop
kill $(lsof -t -i:18000) 2>/dev/null || true

echo ""
echo "========================================================================"
echo "🔍 Analyzing Rollout Traces & Verifying Smoke Criteria..."
echo "========================================================================"
PYTHONPATH=. ./.venv/bin/python scripts/verify_rl_smoke_behavior.py "$OUT_BASE"

echo ""
echo "========================================================================"
echo "🎯 Executing 1 GRPO Update Step on cis_gpu1.utlab.ltd..."
echo "========================================================================"

# Sync traces to GPU1
echo "Syncing traces to cis_gpu1.utlab.ltd..."
ssh cis_gpu1.utlab.ltd "mkdir -p /data/home/huangzixuan/MARBLE/$OUT_BASE"
rsync -avz "$OUT_BASE/" "cis_gpu1.utlab.ltd:/data/home/huangzixuan/MARBLE/$OUT_BASE/"

# Run GRPO update on remote GPU
ssh cis_gpu1.utlab.ltd bash -s "$OUT_BASE" "$CKPT" << 'REMOTE_GRPO'
set -euo pipefail
REMOTE_OUT_BASE="$1"
REMOTE_INIT_CKPT="$2"
cd /data/home/huangzixuan/MARBLE
source /data/home/huangzixuan/miniconda3/bin/activate sigir 2>/dev/null || true

OUT_CKPT="runs/checkpoints/qwen_rl_v2_smoke"
mkdir -p "$OUT_CKPT"

echo "Collecting traces and rewards on GPU1 from $REMOTE_OUT_BASE..."
TRACES=()
REWARDS=()
while IFS= read -r summary_file; do
    trace_file="${summary_file%summary.json}memory_trace.jsonl"
    if [ -f "$trace_file" ]; then
        TRACES+=("$trace_file")
        REWARDS+=("$summary_file")
    fi
done < <(find "$REMOTE_OUT_BASE" -name "summary.json")

echo "Found ${#TRACES[@]} traces for GRPO update."

python -m marble.experiments.train_controller \
    --mode qwen_rl \
    --traces "${TRACES[@]}" \
    --rewards "${REWARDS[@]}" \
    --init "$REMOTE_INIT_CKPT" \
    --base-model /data/home/huangzixuan/models/Qwen3.5-4B \
    --epochs 1 \
    --max-len 4096 \
    --out "$OUT_CKPT"

echo "✅ 1 GRPO UPDATE COMPLETED! Output saved to $OUT_CKPT."
REMOTE_GRPO

# Sync updated checkpoint back to local
echo "Syncing updated checkpoint back to local repository..."
rsync -avz "cis_gpu1.utlab.ltd:/data/home/huangzixuan/MARBLE/runs/checkpoints/qwen_rl_v2_smoke" "runs/checkpoints/"

echo ""
echo "========================================================================"
echo "🎉 6-TASK RL SMOKE CHECK & 1 GRPO UPDATE FULLY COMPLETED!"
echo "========================================================================"
