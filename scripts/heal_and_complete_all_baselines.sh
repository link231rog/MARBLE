#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Baseline Self-Healing and Completion Suite
# 
# 1. Reruns timed-out/failed tasks in existing baseline directories:
#    - mem0_style (Research 18, 59)
#    - no_memory (Database 56, 59; Research 59)
#    - g_memory_style (Database 55, 62; Coding 14, 31, 41, 80)
# 2. Executes complete 24-task Hard Benchmark (test_hard) for missing baselines:
#    - collabmem_style (24 tasks)
#    - copper_style (24 tasks)
# 3. Automatically re-aggregates Master Table 1 into:
#    - runs/master_table_final_hard.json
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_FILE="runs/heal_and_complete.log"
mkdir -p runs
touch "$LOG_FILE"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "========================================================================"
log "🚀 STARTING MARBLE BASELINE RE-RUN & AUTO-HEALING PIPELINE"
log "========================================================================"

# Helper function to clean docker
clean_docker() {
    log "🧹 Cleaning dangling containers and volumes..."
    [ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
    docker volume prune -f >/dev/null 2>&1 || true
    sleep 3
}

# ------------------------------------------------------------------------------
# STEP 1: Heal Mem0 (Re-run incomplete tasks in existing directory)
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "⚡ [Step 1/5] Healing Mem0 Baseline..."
log "========================================================================"
MEM0_DIR="runs/hard_mem0_style_test_hard_c24_20260907_133220"
if [ -d "$MEM0_DIR" ]; then
    ./scripts/run_hard_benchmark.sh \
        --baseline mem0_style \
        --split test_hard \
        --concurrency 4 \
        --task-timeout 0 \
        --out "$MEM0_DIR" 2>&1 | tee -a "$LOG_FILE" || true
    clean_docker
fi

# ------------------------------------------------------------------------------
# STEP 2: Heal No-Memory (Re-run incomplete tasks in existing directory)
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "⚡ [Step 2/5] Healing No-Memory Baseline..."
log "========================================================================"
NOMEM_DIR="runs/hard_no_memory_test_hard_c24_20260907_115659"
if [ -d "$NOMEM_DIR" ]; then
    ./scripts/run_hard_benchmark.sh \
        --baseline no_memory \
        --split test_hard \
        --concurrency 4 \
        --task-timeout 0 \
        --out "$NOMEM_DIR" 2>&1 | tee -a "$LOG_FILE" || true
    clean_docker
fi

# ------------------------------------------------------------------------------
# STEP 3: Heal G-Memory (Re-run incomplete tasks in existing directory)
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "⚡ [Step 3/5] Healing G-Memory Baseline (24 Concurrency)..."
log "========================================================================"
GMEM_DIR="runs/hard_g_memory_style_test_hard_c24_20260907_183224"
if [ -d "$GMEM_DIR" ]; then
    ./scripts/run_hard_benchmark.sh \
        --baseline g_memory_style \
        --split test_hard \
        --concurrency 24 \
        --port-offset 10 \
        --task-timeout 0 \
        --out "$GMEM_DIR" 2>&1 | tee -a "$LOG_FILE" || true
    clean_docker
fi

# ------------------------------------------------------------------------------
# STEP 4: Full 24-Task Execution for CollabMem
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "⚡ [Step 4/5] Running Full 24-Task Benchmark for CollabMem (24 Concurrency)..."
log "========================================================================"
COLLAB_DIR=""
for r in runs/hard_collabmem_style_test_hard_c24_*; do
    if [ -d "$r" ]; then
        ok_cnt=$(find "$r" -name "summary.json" -exec grep -l '"status": "ok"' {} + 2>/dev/null | wc -l || true)
        if [ "$ok_cnt" -ge 24 ]; then
            COLLAB_DIR="$r"
            break
        fi
    fi
done

if [ -n "$COLLAB_DIR" ]; then
    log "✅ CollabMem already completed (24/24 ok) in $COLLAB_DIR. Skipping."
else
    COLLAB_NEW="runs/hard_collabmem_style_test_hard_c24_$(date +%Y%m%d_%H%M%S)"
    ./scripts/run_hard_benchmark.sh \
        --baseline collabmem_style \
        --split test_hard \
        --concurrency 24 \
        --port-offset 10 \
        --task-timeout 0 \
        --out "$COLLAB_NEW" 2>&1 | tee -a "$LOG_FILE"
    clean_docker
fi

# ------------------------------------------------------------------------------
# STEP 5: Full 24-Task Execution for CoPPER
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "⚡ [Step 5/5] Running Full 24-Task Benchmark for CoPPER (24 Concurrency)..."
log "========================================================================"
COPPER_DIR=""
for r in runs/hard_copper_style_test_hard_c24_*; do
    if [ -d "$r" ]; then
        ok_cnt=$(find "$r" -name "summary.json" -exec grep -l '"status": "ok"' {} + 2>/dev/null | wc -l || true)
        if [ "$ok_cnt" -ge 24 ]; then
            COPPER_DIR="$r"
            break
        fi
    fi
done

if [ -n "$COPPER_DIR" ]; then
    log "✅ CoPPER already completed (24/24 ok) in $COPPER_DIR. Skipping."
else
    COPPER_NEW="runs/hard_copper_style_test_hard_c24_$(date +%Y%m%d_%H%M%S)"
    ./scripts/run_hard_benchmark.sh \
        --baseline copper_style \
        --split test_hard \
        --concurrency 24 \
        --port-offset 10 \
        --task-timeout 0 \
        --out "$COPPER_NEW" 2>&1 | tee -a "$LOG_FILE"
    clean_docker
fi

# ------------------------------------------------------------------------------
# FINALIZE: Re-aggregate Master Table 1
# ------------------------------------------------------------------------------
log ""
log "========================================================================"
log "📊 [Finalize] Re-aggregating Master Table 1 across all updated runs..."
log "========================================================================"
./.venv/bin/python scripts/aggregate_hard_master_table.py \
    --run-dir runs/ \
    --manifest configs/experiments/multiagentbench_hard_frozen.json \
    --split test_hard \
    --out-json runs/master_table_final_hard.json 2>&1 | tee -a "$LOG_FILE"

log "========================================================================"
log "🎉 ALL BASELINE HEALING & COMPLETION TASKS FINISHED!"
log "========================================================================"
