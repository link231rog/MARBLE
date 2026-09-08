#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Resume Incomplete Tasks Suite (No Timeout)
# Resumes execution for any unfinished/timed-out tasks in the latest Ours runs
# directly into their respective run directories with TASK_TIMEOUT=0.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

CONCURRENCY="${CONCURRENCY:-24}"
SPLIT="${SPLIT:-test_hard}"
MANIFEST="${MANIFEST:-configs/experiments/multiagentbench_hard_frozen.json}"
LOG_FILE="runs/resume_ours_incomplete.log"

echo "======================================================================"
echo "🚀 MARBLE Resume Incomplete Tasks Suite (No Timeout)"
echo "  Split:       $SPLIT"
echo "  Manifest:    $MANIFEST"
echo "  Concurrency: $CONCURRENCY"
echo "  Time:        $(date)"
echo "======================================================================"

# Clean up any leftover Docker containers
[ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
docker volume prune -f >/dev/null 2>&1 || true

RUN_DIRS=(
    "ours_base:runs/hard_ours_base_test_hard_c24_20260908_002731"
    "ours_sft:runs/hard_ours_sft_test_hard_c24_20260907_235705"
    "ours_rl:runs/hard_ours_rl_test_hard_c24_20260907_232632"
)

for entry in "${RUN_DIRS[@]}"; do
    baseline="${entry%%:*}"
    rdir="${entry##*:}"

    echo ""
    echo "======================================================================"
    echo "⚡ [Resume] Checking baseline: $baseline in $rdir"
    echo "Time: $(date)"
    echo "======================================================================"

    if [ ! -d "$rdir" ]; then
        echo "❌ Directory $rdir does not exist! Skipping."
        continue
    fi

    ok_count=$(find "$rdir" -name "summary.json" -exec grep -l '"status": "ok"' {} + 2>/dev/null | wc -l || true)
    echo "Current status: $ok_count/24 tasks completed with status: ok."

    if [ "$ok_count" -ge 24 ]; then
        echo "✅ All 24 tasks in $rdir are already status: ok! Skipping."
        continue
    fi

    echo "⚡ Launching run_hard_benchmark to resume remaining tasks in $rdir (TASK_TIMEOUT=0)..."

    EXTRA_FLAGS=""
    case "$baseline" in
        ours_rl)
            rl_ckpt=$(ls -td runs/checkpoints/qwen_rl runs/qwen_rl runs/*/checkpoints/qwen_rl 2>/dev/null | head -n 1 || true)
            if [ -n "$rl_ckpt" ] && [ -d "$rl_ckpt" ]; then
                EXTRA_FLAGS="--controller-checkpoint $rl_ckpt"
            elif [ -n "${MARBLE_QWEN_RL_MODEL:-}" ]; then
                EXTRA_FLAGS="--qwen-api-model $MARBLE_QWEN_RL_MODEL"
            else
                echo "❌ FATAL: Baseline 'ours_rl' requires a trained checkpoint (runs/checkpoints/qwen_rl) or MARBLE_QWEN_RL_MODEL!" >&2
                echo "Cannot proceed with silent fallback to untrained base model." >&2
                exit 1
            fi
            ;;
        ours_sft)
            sft_ckpt=$(ls -td runs/checkpoints/qwen_sft runs/qwen_sft runs/*/checkpoints/qwen_sft 2>/dev/null | head -n 1 || true)
            if [ -n "$sft_ckpt" ] && [ -d "$sft_ckpt" ]; then
                EXTRA_FLAGS="--controller-checkpoint $sft_ckpt"
            elif [ -n "${MARBLE_QWEN_SFT_MODEL:-}" ]; then
                EXTRA_FLAGS="--qwen-api-model $MARBLE_QWEN_SFT_MODEL"
            else
                echo "❌ FATAL: Baseline 'ours_sft' requires a trained checkpoint (runs/checkpoints/qwen_sft) or MARBLE_QWEN_SFT_MODEL!" >&2
                echo "Cannot proceed with silent fallback to untrained base model." >&2
                exit 1
            fi
            ;;
    esac

    TASK_TIMEOUT=0 ./scripts/run_hard_benchmark.sh \
        --baseline "$baseline" \
        --split "$SPLIT" \
        --manifest "$MANIFEST" \
        --concurrency "$CONCURRENCY" \
        --out "$rdir" \
        $EXTRA_FLAGS

    [ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
    docker volume prune -f >/dev/null 2>&1 || true
    sleep 3

    new_ok=$(find "$rdir" -name "summary.json" -exec grep -l '"status": "ok"' {} + 2>/dev/null | wc -l || true)
    echo "After resume: $new_ok/24 tasks completed with status: ok in $rdir."
done

echo ""
echo "======================================================================"
echo "🎉 All Resumed Runs Finished! Re-aggregating Master Table 1..."
echo "Time: $(date)"
echo "======================================================================"

python3 scripts/aggregate_hard_master_table.py \
    --run-dir runs/ \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out-json "runs/master_table_ours_methods.json"

echo "======================================================================"
echo "✅ Results saved to runs/master_table_ours_methods.json"
echo "======================================================================"
