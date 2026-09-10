#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Controller Parameter Scaling & Capacity Sensitivity Pipeline
# Scales the Meta-Memory Controller across 0.8B, 2B, 4B, and 9B architectures.
# Evaluates 3 paradigms per scale: Ours-Base, Ours-SFT, and Ours-GRPO.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SCALES="0.8b 2b 4b 9b"
METHODS="ours_base ours_sft ours_rl"
MANIFEST="configs/experiments/rl_smoke_train_6tasks.json"
SPLIT="train_hard"
CONCURRENCY=6
SEEDS="42"
OUT_BASE="runs/controller_scaling_analysis"
SKIP_TRAIN=false
SKIP_EVAL=false
FORCE_RETRAIN=false
REMOTE_HOST="cis_gpu1.utlab.ltd"
REMOTE_REPO="/data/home/huangzixuan/MARBLE"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scales)
            SCALES="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --seed|--seeds)
            SEEDS="$2"
            shift 2
            ;;
        --out)
            OUT_BASE="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUT_BASE"

echo "========================================================================"
echo "📈 MARBLE Controller Capacity Scaling Suite (Parameter Sensitivity)"
echo "  Scales:      $SCALES"
echo "  Methods:     $METHODS"
echo "  Manifest:    $MANIFEST"
echo "  Split:       $SPLIT"
echo "  Concurrency: $CONCURRENCY"
echo "  Seeds:       $SEEDS"
echo "  Output Dir:  $OUT_BASE"
echo "========================================================================"

get_model_meta() {
    local SCALE="$1"
    case "$SCALE" in
        0.8b|0.8B)
            MODEL_NAME="Qwen/Qwen3.5-0.8B"
            MODEL_DIR="/data/home/huangzixuan/models/Qwen3.5-0.8B"
            SFT_CKPT="runs/checkpoints/controller_scaling/qwen_0.8b_sft"
            RL_CKPT="runs/checkpoints/controller_scaling/qwen_0.8b_rl"
            ;;
        2b|2B)
            MODEL_NAME="Qwen/Qwen3.5-2B"
            MODEL_DIR="/data/home/huangzixuan/models/Qwen3.5-2B"
            SFT_CKPT="runs/checkpoints/controller_scaling/qwen_2b_sft"
            RL_CKPT="runs/checkpoints/controller_scaling/qwen_2b_rl"
            ;;
        4b|4B)
            MODEL_NAME="Qwen/Qwen3.5-4B"
            MODEL_DIR="/data/home/huangzixuan/models/Qwen3.5-4B"
            SFT_CKPT="runs/checkpoints/controller_scaling/qwen_4b_sft"
            RL_CKPT="runs/checkpoints/controller_scaling/qwen_4b_rl"
            ;;
        9b|9B)
            MODEL_NAME="Qwen/Qwen3.5-9B"
            MODEL_DIR="/data/home/huangzixuan/models/Qwen3.5-9B"
            SFT_CKPT="runs/checkpoints/controller_scaling/qwen_9b_sft"
            RL_CKPT="runs/checkpoints/controller_scaling/qwen_9b_rl"
            ;;
        *)
            echo "Unknown scale: $SCALE" >&2
            return 1
            ;;
    esac
}

switch_remote_model() {
    local M_DIR="$1"
    local M_NAME="$2"
    local SCALE="$3"
    echo ""
    echo ">>> [GPU1] Switching served controller to $M_NAME (Scale: $SCALE)..."
    ssh "$REMOTE_HOST" "bash -s" << REMOTE_SWITCH
set -euo pipefail
pkill -9 -f serve_qwen_openai.py 2>/dev/null || true
tmux kill-session -t qwen_server 2>/dev/null || true
for i in \$(seq 1 10); do
    if ! lsof -i :8000 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
tmux new-session -d -s qwen_server "MARBLE_MODEL_PATH=$M_DIR MARBLE_SERVED_MODEL_NAME=$M_NAME MARBLE_SCALE=$SCALE PORT=8000 /data/home/huangzixuan/miniconda3/envs/sigir/bin/python /data/home/huangzixuan/workspace/MARBLE/scripts/serve_qwen_openai.py >> /data/home/huangzixuan/vllm_serve.log 2>&1"
REMOTE_SWITCH
    echo "Waiting for endpoint http://127.0.0.1:18000/v1/models to verify $M_NAME..."
    for i in $(seq 1 45); do
        if curl -s "http://127.0.0.1:18000/v1/models" | grep -q "$M_NAME"; then
            echo ">>> [GPU1] $M_NAME is READY and responding!"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Timeout waiting for $M_NAME on GPU1!"
    return 1
}

wait_for_model_download() {
    local M_DIR="$1"
    local M_NAME="$2"
    echo ">>> [GPU1] Verifying model download for $M_NAME at $M_DIR..."
    ssh "$REMOTE_HOST" "bash -s" << CHECK_DL
set -euo pipefail
if ls "$M_DIR"/*.incomplete >/dev/null 2>&1; then
    echo "Model $M_NAME has ongoing downloads (.incomplete files detected in $M_DIR). Waiting for completion..."
    while ls "$M_DIR"/*.incomplete >/dev/null 2>&1; do
        sleep 15
    done
    echo ">>> All .incomplete files cleared for $M_NAME!"
fi
CHECK_DL
}

train_scale_sft() {
    local SCALE="$1"
    local M_DIR="$2"
    local SFT_OUT="$3"

    if [ "$FORCE_RETRAIN" = false ]; then
        if [ "$SCALE" = "4b" ] && [ -d "$SFT_OUT" ]; then
            echo ">>> [GPU1] Scale 4B SFT checkpoint already exists at $SFT_OUT, skipping."
            return 0
        fi

        if [ "$SCALE" = "4b" ] && [ ! -d "$SFT_OUT" ]; then
            if ssh "$REMOTE_HOST" "[ -d $REMOTE_REPO/runs/checkpoints/qwen_sft ]"; then
                echo ">>> [GPU1] Scale 4B SFT: Initializing isolated copy from runs/checkpoints/qwen_sft to $SFT_OUT..."
                ssh "$REMOTE_HOST" "mkdir -p $REMOTE_REPO/runs/checkpoints/controller_scaling && cp -r $REMOTE_REPO/runs/checkpoints/qwen_sft $REMOTE_REPO/$SFT_OUT"
                mkdir -p "$SFT_OUT"
                rsync -avz "$REMOTE_HOST:$REMOTE_REPO/$SFT_OUT/" "$SFT_OUT/"
                return 0
            fi
        fi

        if [ -d "$SFT_OUT" ]; then
            echo ">>> [GPU1] SFT checkpoint for $SCALE already exists at $SFT_OUT, skipping."
            return 0
        fi
    fi

    echo ""
    echo "===================================================================="
    echo "⚡ [GPU1] Training SFT Adapter for Controller Scale: $SCALE"
    echo "===================================================================="
REMOTE_PYTHON="/data/home/huangzixuan/miniconda3/envs/sigir/bin/python"
    ssh "$REMOTE_HOST" "bash -s" << REMOTE_SFT
set -euo pipefail
cd $REMOTE_REPO
mkdir -p "$SFT_OUT"
DIRS=()
for d in runs/train_hard_traces runs/train_rollout_ours_base runs/rl_smoke_check runs/grpo_recovery_smoke_20260909; do
    [ -d "\$d" ] && DIRS+=("\$d")
done
TRACES=(\$(find "\${DIRS[@]}" -name "memory_trace.jsonl" | head -n 250))
$REMOTE_PYTHON -m marble.experiments.train_controller \
    --mode qwen_sft \
    --traces "\${TRACES[@]}" \
    --base-model "$M_DIR" \
    --epochs 3 \
    --out "$SFT_OUT"
REMOTE_SFT
    mkdir -p "$SFT_OUT"
    rsync -avz "$REMOTE_HOST:$REMOTE_REPO/$SFT_OUT/" "$SFT_OUT/"
    echo ">>> [GPU1] Scale $SCALE SFT training complete and synced to $SFT_OUT."
}

train_scale_grpo() {
    local SCALE="$1"
    local M_DIR="$2"
    local SFT_IN="$3"
    local RL_OUT="$4"

    if [ "$FORCE_RETRAIN" = false ]; then
        if [ "$SCALE" = "4b" ] && [ -d "$RL_OUT" ]; then
            echo ">>> [GPU1] Scale 4B GRPO checkpoint already exists at $RL_OUT, skipping."
            return 0
        fi

        if [ -d "$RL_OUT" ]; then
            echo ">>> [GPU1] GRPO checkpoint for $SCALE already exists at $RL_OUT, skipping."
            return 0
        fi
    fi

    echo ""
    echo "===================================================================="
    echo "🎯 [GPU1] Training 1-Step GRPO Adapter for Controller Scale: $SCALE"
    echo "===================================================================="
    REMOTE_PYTHON="/data/home/huangzixuan/miniconda3/envs/sigir/bin/python"
    ssh "$REMOTE_HOST" "bash -s" << REMOTE_GRPO
set -euo pipefail
cd $REMOTE_REPO
mkdir -p "$RL_OUT"

TRACES=()
REWARDS=()
SOURCE_DIR="runs/grpo_recovery_smoke_20260909"
if [ ! -d "\$SOURCE_DIR" ]; then
    SOURCE_DIR="runs/rl_smoke_check"
fi
while IFS= read -r summary_file; do
    trace_file="\${summary_file%summary.json}memory_trace.jsonl"
    if [ -f "\$trace_file" ]; then
        TRACES+=("\$trace_file")
        REWARDS+=("\$summary_file")
    fi
done < <(find "\$SOURCE_DIR" -name "summary.json")

export MARBLE_RL_CHUNK_SIZE=1
$REMOTE_PYTHON -m marble.experiments.train_controller \
    --mode qwen_rl \
    --traces "\${TRACES[@]}" \
    --rewards "\${REWARDS[@]}" \
    --init "$SFT_IN" \
    --base-model "$M_DIR" \
    --epochs 1 \
    --advantage-mode task_grpo \
    --out "$RL_OUT"
REMOTE_GRPO
    mkdir -p "$RL_OUT"
    rsync -avz "$REMOTE_HOST:$REMOTE_REPO/$RL_OUT/" "$RL_OUT/"
    echo ">>> [GPU1] Scale $SCALE GRPO training complete and synced to $RL_OUT."
}

# Iterate through each scale
for SCALE in $SCALES; do
    get_model_meta "$SCALE"

    echo ""
    echo "===================================================================="
    echo "🔬 PROCESSING CONTROLLER SCALE: $SCALE ($MODEL_NAME)"
    echo "===================================================================="

    # Phase 0: Ensure model download is complete
    wait_for_model_download "$MODEL_DIR" "$MODEL_NAME"

    # Phase 1: Train SFT & GRPO adapters on GPU1 if needed
    if [ "$SKIP_TRAIN" = false ]; then
        train_scale_sft "$SCALE" "$MODEL_DIR" "$SFT_CKPT"
        train_scale_grpo "$SCALE" "$MODEL_DIR" "$SFT_CKPT" "$RL_CKPT"
    fi

    if [ "$SKIP_EVAL" = true ]; then
        echo ">>> Skipping evaluation as requested (--skip-eval)."
        continue
    fi

    # Phase 2: Switch remote serving model to current scale
    switch_remote_model "$MODEL_DIR" "$MODEL_NAME" "$SCALE"

    # Phase 3: Evaluate each requested method and seed
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            EVAL_DIR="$OUT_BASE/${SCALE}/${METHOD}/seed_${SEED}"
            mkdir -p "$EVAL_DIR"

            echo ""
            echo ">>> [Evaluation] Scale=$SCALE | Method=$METHOD | Seed=$SEED | Concurrency=$CONCURRENCY"

            EXTRA_ARGS=(--qwen-api-base "http://127.0.0.1:18000/v1")
            if [ "$METHOD" = "ours_base" ]; then
                EXTRA_ARGS+=(--qwen-api-model "$MODEL_NAME")
            elif [ "$METHOD" = "ours_sft" ]; then
                EXTRA_ARGS+=(--controller-checkpoint "$SFT_CKPT" --qwen-api-model "qwen_${SCALE}_sft")
            elif [ "$METHOD" = "ours_rl" ]; then
                EXTRA_ARGS+=(--controller-checkpoint "$RL_CKPT" --qwen-api-model "qwen_${SCALE}_rl")
            fi

            TASK_TIMEOUT=0 ./scripts/run_hard_benchmark_pool.sh \
                --baseline "$METHOD" \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --concurrency "$CONCURRENCY" \
                --seed "$SEED" \
                "${EXTRA_ARGS[@]}" \
                --out "$EVAL_DIR"
        done
    done
done

echo ""
echo "========================================================================"
echo "🎉 CONTROLLER SCALING EXPERIMENT SUITE COMPLETED!"
echo "========================================================================"
echo "📊 Aggregating Results Table Across Scales & Methods..."
echo "========================================================================"

./.venv/bin/python scripts/aggregate_scaling_analysis_table.py     --base-dir "$OUT_BASE"     --scales $SCALES     --methods $METHODS     --json-out "$OUT_BASE/scaling_analysis_report.json"

echo "✅ All scaling results aggregated at: $OUT_BASE/scaling_analysis_report.json"
