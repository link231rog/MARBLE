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
MANIFEST="configs/experiments/multiagentbench_stratified_frozen.json"

echo "========================================================"
echo "[Stream 2 - Research] Starting Research Benchmark Stream"
echo "Concurrency: 2 Workers (Parallel Task Dispatch)"
echo "Target Dir: $OUT_DIR"
echo "Manifest: $MANIFEST"
echo "Worker Model: $MARBLE_WORKER_MODEL"
echo "========================================================"

BASELINES=(
    "no_memory"
    "single_agent"
    "global_add_all"
    "lts_style"
    "memory_r1_style"
)

# Helper function to run a slice of tasks on a worker
run_research_slice() {
    local worker_id="$1"
    local baseline="$2"
    local split="$3"
    local task_ids="$4"
    echo "[Worker $worker_id] Starting: baseline=$baseline split=$split tasks=$task_ids"
    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark research \
        --manifest "$MANIFEST" \
        --split "$split" \
        --task-ids "$task_ids" \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_${split}_${baseline}_w${worker_id}.log"
}

# Step 1: Run Train Splits (Standard: 1, 3, 4 | Hard: 10, 11, 15)
echo ">>> [Stream 2.1] Research Train Runs (2-Worker Parallel)..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> [Research] Train: Baseline = $baseline (2 Workers)"
    echo "--------------------------------------------------------"
    # Worker 1: 1, 4, 11
    run_research_slice 1 "$baseline" "train" "1,4,11" &
    PID1=$!
    # Worker 2: 3, 10, 15
    run_research_slice 2 "$baseline" "train" "3,10,15" &
    PID2=$!
    wait $PID1 $PID2
done

# Step 2: Run Test Splits (Standard: 2, 5, 6, 7 | Hard: 18, 20, 26, 51)
echo ">>> [Stream 2.2] Research Test Runs (2-Worker Parallel)..."
for baseline in "${BASELINES[@]}"; do
    echo "--------------------------------------------------------"
    echo ">>> [Research] Test: Baseline = $baseline (2 Workers)"
    echo "--------------------------------------------------------"
    # Worker 1: 2, 6, 18, 26
    run_research_slice 1 "$baseline" "test" "2,6,18,26" &
    PID1=$!
    # Worker 2: 5, 7, 20, 51
    run_research_slice 2 "$baseline" "test" "5,7,20,51" &
    PID2=$!
    wait $PID1 $PID2
done

echo "========================================================"
echo "[Stream 2 - Research] All Stratified Research Tasks Completed!"
echo "========================================================"
