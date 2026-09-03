#!/bin/bash
set -u

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
echo "[Stream 1 - Database] Starting Database Benchmark Stream"
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

# Step 1: Run Train Split (Tasks 1, 2, 3)
echo ">>> [Stream 1.1] Database Train Split (Tasks 1, 2, 3)..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> [Database] Train Split: Baseline = $baseline"
    echo "--------------------------------------------------------"
    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark database \
        --manifest configs/experiments/multiagentbench_frozen.json \
        --split train \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_train_${baseline}.log"
done

echo ">>> [Stream 1.1] Database Train Split Completed!"

# Step 2: Run Test Split (Tasks 4..12)
echo ">>> [Stream 1.2] Database Test Split (Tasks 4..12)..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> [Database] Test Split: Baseline = $baseline"
    echo "--------------------------------------------------------"
    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark database \
        --manifest configs/experiments/multiagentbench_frozen.json \
        --split test \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_test_${baseline}.log"
done

echo "========================================================"
echo "[Stream 1 - Database] All Database Tasks Completed!"
echo "========================================================"
