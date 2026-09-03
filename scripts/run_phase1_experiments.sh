#!/bin/bash
set -u

# 1. Environment and Network Hygiene
unset NO_PROXY no_proxy
REPO_DIR="/Users/huangzixuan/Documents/404 not found/Good Night/Research/MAS&Memory/code/MARBLE"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
OUT_DIR="runs/nvidia-gpt-oss-main-20260903"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo "[Phase 1] Starting Autonomous Baseline Evaluation Suite"
echo "Target Dir: $OUT_DIR"
echo "Worker Model: $MARBLE_WORKER_MODEL"
echo "========================================================"

BASELINES=(
    "no_memory"
    "single_agent"
    "global_add_all"
    "lts_style"
    "memory_r1_style"
)

# Step 1: Run 6 Train Tasks first to collect trajectory data for Controller SFT & RL training
echo ">>> [Phase 1.1] Running Train Split (6 Tasks) to collect Trajectories..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> Running Train Split: Baseline = $baseline"
    echo "--------------------------------------------------------"
    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark database,research \
        --manifest configs/experiments/multiagentbench_frozen.json \
        --split train \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/train_${baseline}.log"
done

echo ">>> [Phase 1.1] Completed! All 6 Train Tasks finished across 5 baselines."
echo ">>> Trajectories are ready for Controller (Qwen3-4B) SFT / RL post-training."

# Step 2: Run Test Split (18 Tasks) across all 5 Baselines
echo ">>> [Phase 1.2] Running Test Split (18 Tasks) across 5 Baselines..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> Running Test Split: Baseline = $baseline"
    echo "--------------------------------------------------------"
    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark database,research \
        --manifest configs/experiments/multiagentbench_frozen.json \
        --split test \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/test_${baseline}.log"
done

echo "========================================================"
echo "[Phase 1] All Baseline Evaluations Successfully Finished!"
echo "========================================================"
