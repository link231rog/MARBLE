#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MANIFEST="configs/experiments/rl_smoke_train_6tasks.json"
OUT_BASE="runs/rl_smoke_check"
mkdir -p "$OUT_BASE"

echo "========================================================================"
echo "🚀 RL Smoke Verification: 6 Tasks x 4 Rollouts (Seeds 42, 43, 44, 45)"
echo "  Manifest:    $MANIFEST"
echo "  Output Base: $OUT_BASE"
echo "========================================================================"

SEEDS=(42 43 44 45)
for SEED in "${SEEDS[@]}"; do
    SEED_OUT="$OUT_BASE/seed_${SEED}"
    echo ""
    echo ">>> [Rollout Seed $SEED] Dispatching 6 tasks (concurrency 6) to $SEED_OUT..."
    TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
        --baseline ours_sft \
        --manifest "$MANIFEST" \
        --split train_hard \
        --concurrency 6 \
        --seed "$SEED" \
        --controller-checkpoint runs/checkpoints/qwen_sft \
        --qwen-api-model qwen_sft \
        --out "$SEED_OUT"
    echo ">>> [Rollout Seed $SEED] Finished."
done

echo ""
echo "========================================================================"
echo "🔍 Analyzing Rollout Traces & Verifying Smoke Criteria..."
echo "========================================================================"
PYTHONPATH=. ./.venv/bin/python scripts/verify_rl_smoke_behavior.py

echo ""
echo "========================================================================"
echo "🎯 Executing 1 GRPO Update Step on cis_gpu1.utlab.ltd..."
echo "========================================================================"

# Sync traces to GPU1
echo "Syncing traces to cis_gpu1.utlab.ltd..."
ssh cis_gpu1.utlab.ltd "mkdir -p /data/home/huangzixuan/MARBLE/runs/rl_smoke_check"
rsync -avz "$OUT_BASE/" "cis_gpu1.utlab.ltd:/data/home/huangzixuan/MARBLE/runs/rl_smoke_check/"

# Run GRPO update on remote GPU
ssh cis_gpu1.utlab.ltd 'bash -s' << 'REMOTE_GRPO'
set -euo pipefail
cd /data/home/huangzixuan/MARBLE
source /data/home/huangzixuan/MARBLE/.venv/bin/activate 2>/dev/null || true

OUT_CKPT="runs/checkpoints/qwen_rl_smoke"
mkdir -p "$OUT_CKPT"

echo "Collecting traces and rewards on GPU1..."
TRACES=()
REWARDS=()
while IFS= read -r summary_file; do
    trace_file="${summary_file%summary.json}memory_trace.jsonl"
    if [ -f "$trace_file" ]; then
        TRACES+=("$trace_file")
        REWARDS+=("$summary_file")
    fi
done < <(find runs/rl_smoke_check -name "summary.json")

echo "Found ${#TRACES[@]} traces for GRPO update."

python3 -m marble.experiments.train_controller \
    --mode qwen_rl \
    --traces "${TRACES[@]}" \
    --rewards "${REWARDS[@]}" \
    --init runs/checkpoints/qwen_sft \
    --base-model /data/home/huangzixuan/models/Qwen3.5-4B \
    --epochs 1 \
    --out "$OUT_CKPT"

echo "✅ 1 GRPO UPDATE COMPLETED! Output saved to $OUT_CKPT."
REMOTE_GRPO

# Sync updated checkpoint back to local
echo "Syncing updated checkpoint back to local repository..."
rsync -avz "cis_gpu1.utlab.ltd:/data/home/huangzixuan/MARBLE/runs/checkpoints/qwen_rl_smoke" "runs/checkpoints/"

echo ""
echo "========================================================================"
echo "🎉 6-TASK RL SMOKE CHECK & 1 GRPO UPDATE FULLY COMPLETED!"
echo "========================================================================"
