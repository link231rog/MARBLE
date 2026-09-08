#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Table 1: Ours Methods Evaluation Suite (Decoupled)
# Executes ONLY the Ours family methods across 24 Hard Episodes:
#   1. ours_base (Zero-shot prompted memory sharing)
#   2. ours_sft  (SFT controller policy)
#   3. ours_rl   (RL-governed controller policy)
#   4. ours_private_to_global (Ablation: direct sharing without private staging)
#
# Use this script after completing fine-tuning/training for the controller policy.
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

if [ $# -gt 0 ]; then
    BASELINES=("$@")
else
    BASELINES=(
        "ours_rl"
        "ours_sft"
        "ours_base"
        "ours_private_to_global"
    )
fi

echo "======================================================================"
echo "🚀 MARBLE Ours Methods Benchmark Suite (Decoupled)"
echo "  Split:       $SPLIT (24 Episodes: 8 DB + 8 Research + 8 Coding)"
echo "  Manifest:    $MANIFEST"
echo "  Concurrency: $CONCURRENCY"
echo "  Methods (${#BASELINES[@]}): ${BASELINES[*]}"
echo "======================================================================"

for i in "${!BASELINES[@]}"; do
    b="${BASELINES[$i]}"
    echo ""
    echo "======================================================================"
    echo "[$((i+1))/${#BASELINES[@]}] ⚡ Checking Method: $b"
    echo "Time: $(date)"
    echo "======================================================================"

    done_run=""
    target_worker="gpt-oss-20b"
    for r in runs/hard_${b}_${SPLIT}_c${CONCURRENCY}_*; do
        if [ -d "$r" ]; then
            ok_count=$(find "$r" -name "summary.json" -exec grep -l '"status": "ok"' {} + 2>/dev/null | wc -l || true)
            if [ "$ok_count" -ge 24 ]; then
                first_sum=$(find "$r" -name "summary.json" 2>/dev/null | head -n 1)
                if [ -n "$first_sum" ] && grep -q "$target_worker" "$first_sum"; then
                    done_run="$r"
                    break
                fi
            fi
        fi
    done

    if [ -n "$done_run" ]; then
        echo "✅ Method '$b' already has 24/24 completed episodes in $done_run (matching $target_worker). Skipping to next!"
        continue
    fi

    EXTRA_FLAGS=""
    case "$b" in
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
        ours_private_to_global)
            rl_ckpt=$(ls -td runs/checkpoints/qwen_rl runs/qwen_rl runs/*/checkpoints/qwen_rl 2>/dev/null | head -n 1 || true)
            if [ -n "$rl_ckpt" ] && [ -d "$rl_ckpt" ]; then
                EXTRA_FLAGS="--controller-checkpoint $rl_ckpt --ablation visibility:private_to_global"
            elif [ -n "${MARBLE_QWEN_RL_MODEL:-}" ]; then
                EXTRA_FLAGS="--qwen-api-model $MARBLE_QWEN_RL_MODEL --ablation visibility:private_to_global"
            else
                echo "❌ FATAL: Baseline 'ours_private_to_global' requires a trained checkpoint (runs/checkpoints/qwen_rl) or MARBLE_QWEN_RL_MODEL!" >&2
                echo "Cannot proceed with silent fallback to untrained base model." >&2
                exit 1
            fi
            ;;
        ours_linear_rl|learned_controller)
            linear_rl=$(ls -t runs/reproducible_suite_*/checkpoints/rl_policy.json runs/rl_policy.json runs/ours_rl_policy.json 2>/dev/null | head -n 1 || true)
            [ -n "$linear_rl" ] && [ -f "$linear_rl" ] && EXTRA_FLAGS="--controller-checkpoint $linear_rl"
            ;;
        ours_linear_sft)
            linear_sft=$(ls -t runs/reproducible_suite_*/checkpoints/sft_policy.json runs/sft_policy.json runs/ours_sft_policy.json 2>/dev/null | head -n 1 || true)
            [ -n "$linear_sft" ] && [ -f "$linear_sft" ] && EXTRA_FLAGS="--controller-checkpoint $linear_sft"
            ;;
    esac

    echo "⚡ Launching Method: $b..."
    ./scripts/run_hard_benchmark.sh \
        --baseline "$b" \
        --split "$SPLIT" \
        --manifest "$MANIFEST" \
        --concurrency "$CONCURRENCY" \
        $EXTRA_FLAGS

    [ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
    docker volume prune -f >/dev/null 2>&1 || true
    sleep 3
done

echo ""
echo "======================================================================"
echo "🎉 All Ours Runs Complete! Aggregating Results..."
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
