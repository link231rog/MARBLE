#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE 8-Concurrency Parallel Benchmark Runner
# Dispatches MultiAgentBench tasks across up to 8 sandboxed parallel workers.
# Automatically allocates unique ports and Docker compose projects per worker.
# Supports multi-API key rotation to prevent rate-limit bottlenecks.
# ==============================================================================

unset NO_PROXY no_proxy
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
MANIFEST="configs/experiments/multiagentbench_stratified_frozen.json"

# Default arguments
BASELINE="ours_base"
SPLIT="test"
BENCHMARK_TARGET="both"
MAX_CARDS=5
MAX_ITERATIONS=5
CONCURRENCY=8
ENABLE_COMM_GOV=""
DRY_RUN=""
OUT_DIR=""
CONTROLLER_CHECKPOINT=""
LAMBDA="0.15"
BETA="0.25"
ABLATION=""

PORT_OFFSET=0

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--baseline)
            BASELINE="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_TARGET="$2"
            shift 2
            ;;
        -k|--max-cards)
            MAX_CARDS="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --controller-checkpoint)
            CONTROLLER_CHECKPOINT="$2"
            shift 2
            ;;
        --ablation)
            ABLATION="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --port-offset)
            PORT_OFFSET="$2"
            shift 2
            ;;
        --enable-comm-governor)
            ENABLE_COMM_GOV="--enable-comm-governor"
            shift 1
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift 1
            ;;
        -m|--manifest)
            MANIFEST="$2"
            shift 2
            ;;
        -o|--out)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="runs/parallel_${BASELINE}_${SPLIT}_c${CONCURRENCY}_${TIMESTAMP}"
fi
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "⚡ MARBLE High-Concurrency Parallel Runner ($CONCURRENCY Workers)"
echo "  Baseline:    $BASELINE"
echo "  Split:       $SPLIT"
echo "  Benchmark:   $BENCHMARK_TARGET"
echo "  Max Cards:   $MAX_CARDS"
echo "  Port Offset: $PORT_OFFSET"
echo "  Output Dir:  $OUT_DIR"
[ -n "$CONTROLLER_CHECKPOINT" ] && echo "  Checkpoint:  $CONTROLLER_CHECKPOINT"
echo "======================================================================"

# Parse API keys and endpoints for multi-provider rotation
IFS=',' read -r -a QWEN_KEYS <<< "${MARBLE_QWEN_API_KEYS:-${MARBLE_QWEN_API_KEY:-}}"
IFS=',' read -r -a QWEN_BASES <<< "${MARBLE_QWEN_API_BASES:-${MARBLE_QWEN_API_BASE:-}}"
IFS=',' read -r -a QWEN_MODELS <<< "${MARBLE_QWEN_API_MODELS:-${MARBLE_QWEN_API_MODEL:-}}"
IFS=',' read -r -a WORKER_KEYS <<< "${MARBLE_WORKER_API_KEYS:-${NVAPI_KEY:-${OPENAI_API_KEY:-}}}"

get_qwen_key() {
    local idx=$1
    if [ ${#QWEN_KEYS[@]} -gt 0 ]; then
        echo "${QWEN_KEYS[$((idx % ${#QWEN_KEYS[@]}))]}"
    else
        echo ""
    fi
}

get_qwen_base() {
    local idx=$1
    if [ ${#QWEN_BASES[@]} -gt 0 ]; then
        echo "${QWEN_BASES[$((idx % ${#QWEN_BASES[@]}))]}"
    else
        echo ""
    fi
}

get_qwen_model() {
    local idx=$1
    if [ ${#QWEN_MODELS[@]} -gt 0 ]; then
        echo "${QWEN_MODELS[$((idx % ${#QWEN_MODELS[@]}))]}"
    else
        echo ""
    fi
}

get_worker_key() {
    local idx=$1
    if [ ${#WORKER_KEYS[@]} -gt 0 ]; then
        echo "${WORKER_KEYS[$((idx % ${#WORKER_KEYS[@]}))]}"
    else
        echo ""
    fi
}

# Determine Task IDs dynamically from frozen manifest based on Benchmark and Split
read -r -a DB_TASKS <<< "$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='database'))")"
read -r -a RESEARCH_TASKS <<< "$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='research'))")"

if [ ${#DB_TASKS[@]} -eq 0 ] && [ ${#RESEARCH_TASKS[@]} -eq 0 ]; then
    echo "Error: No tasks found in manifest '$MANIFEST' for split '$SPLIT'"
    exit 1
fi
echo "Loaded Tasks from Manifest:"
echo "  Database Tasks: ${DB_TASKS[*]}"
echo "  Research Tasks: ${RESEARCH_TASKS[*]}"

# ------------------------------------------------------------------------------
# Task Dispatcher Helpers
# ------------------------------------------------------------------------------
launch_db_worker() {
    local task_id="$1"
    local worker_id="$2"
    local db_port=$((54320 + PORT_OFFSET + worker_id))
    local prom_port=$((55000 + PORT_OFFSET + worker_id))
    local node_port=$((56000 + PORT_OFFSET + worker_id))
    local pg_exp_port=$((57000 + PORT_OFFSET + worker_id))
    local compose_proj="marble_db_${BASELINE}_w${worker_id}"

    local q_key="$(get_qwen_key "$worker_id")"
    local q_base="$(get_qwen_base "$worker_id")"
    local q_model="$(get_qwen_model "$worker_id")"
    local w_key="$(get_worker_key "$worker_id")"

    (
        export MARBLE_DB_PORT="$db_port"
        export MARBLE_PROM_PORT="$prom_port"
        export MARBLE_NODE_PORT="$node_port"
        export MARBLE_PG_EXPORTER_PORT="$pg_exp_port"
        export MARBLE_COMPOSE_PROJECT="$compose_proj"
        [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
        [ -n "$q_base" ] && export MARBLE_QWEN_API_BASE="$q_base"
        [ -n "$q_model" ] && export MARBLE_QWEN_API_MODEL="$q_model"
        [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"

        echo "[DB Worker $worker_id] Launching Database Task $task_id (Port: $db_port)..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $UV run python -u -m marble.experiments.run_benchmark --benchmark database --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            $UV run python -u -m marble.experiments.run_benchmark \
                --benchmark database \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --task-ids "$task_id" \
                --baseline "$BASELINE" \
                --seed 42 \
                --max-iterations "$MAX_ITERATIONS" \
                --retrieval visible_k \
                --max-cards "$MAX_CARDS" \
                --lambda "$LAMBDA" \
                --beta "$BETA" \
                --task-timeout 900 \
                $EXTRA_ARGS \
                --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_task_${task_id}_w${worker_id}.log"

            docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true
        fi
        echo "[DB Worker $worker_id] Finished Database Task $task_id."
    ) &
}

launch_research_worker() {
    local task_id="$1"
    local worker_id="$2"
    local q_key="$(get_qwen_key "$worker_id")"
    local q_base="$(get_qwen_base "$worker_id")"
    local q_model="$(get_qwen_model "$worker_id")"
    local w_key="$(get_worker_key "$worker_id")"

    (
        [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
        [ -n "$q_base" ] && export MARBLE_QWEN_API_BASE="$q_base"
        [ -n "$q_model" ] && export MARBLE_QWEN_API_MODEL="$q_model"
        [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"

        echo "[Research Worker $worker_id] Launching Research Task $task_id..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $UV run python -u -m marble.experiments.run_benchmark --benchmark research --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            $UV run python -u -m marble.experiments.run_benchmark \
                --benchmark research \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --task-ids "$task_id" \
                --baseline "$BASELINE" \
                --seed 42 \
                --max-iterations "$MAX_ITERATIONS" \
                --retrieval visible_k \
                --max-cards "$MAX_CARDS" \
                --lambda "$LAMBDA" \
                --beta "$BETA" \
                --task-timeout 900 \
                $EXTRA_ARGS \
                --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_task_${task_id}_w${worker_id}.log"
        fi
        echo "[Research Worker $worker_id] Finished Research Task $task_id."
    ) &
}

# ------------------------------------------------------------------------------
# Execution Flow
# ------------------------------------------------------------------------------
if [ "$BENCHMARK_TARGET" == "both" ] && [ "$CONCURRENCY" -ge 16 ]; then
    echo "======================================================================"
    echo "🚀 [16-Concurrency Simultaneous Mode] Launching Database & Research in Parallel!"
    echo "  Database Workers: 0..$(( ${#DB_TASKS[@]} - 1 ))"
    echo "  Research Workers: ${#DB_TASKS[@]}..$(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} - 1 ))"
    echo "======================================================================"
    PIDS=()
    for i in "${!DB_TASKS[@]}"; do
        launch_db_worker "${DB_TASKS[$i]}" "$i"
        PIDS+=($!)
    done
    for j in "${!RESEARCH_TASKS[@]}"; do
        launch_research_worker "${RESEARCH_TASKS[$j]}" "$(( ${#DB_TASKS[@]} + j ))"
        PIDS+=($!)
    done

    echo ">>> All ${#PIDS[@]} workers active in parallel! Waiting for full wave completion..."
    wait "${PIDS[@]}"
    echo ">>> All Database and Research Tasks successfully completed in parallel!"

else
    # Phased batch mode
    if [ "$BENCHMARK_TARGET" == "both" ] || [ "$BENCHMARK_TARGET" == "database" ]; then
        echo ">>> [Phase 1/2] Launching Database Benchmark across $CONCURRENCY parallel sandboxes..."
        PIDS=()
        for i in "${!DB_TASKS[@]}"; do
            launch_db_worker "${DB_TASKS[$i]}" "$((i % CONCURRENCY))"
            PIDS+=($!)
            if [ $(( (i + 1) % CONCURRENCY )) -eq 0 ] && [ $((i + 1)) -lt ${#DB_TASKS[@]} ]; then
                echo "Waiting for current Database batch to complete..."
                wait "${PIDS[@]}"
                PIDS=()
            fi
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        echo ">>> Database Tasks successfully completed!"
    fi

    if [ "$BENCHMARK_TARGET" == "both" ] || [ "$BENCHMARK_TARGET" == "research" ]; then
        echo ">>> [Phase 2/2] Launching Research Benchmark across $CONCURRENCY parallel workers..."
        PIDS=()
        for i in "${!RESEARCH_TASKS[@]}"; do
            launch_research_worker "${RESEARCH_TASKS[$i]}" "$((i % CONCURRENCY))"
            PIDS+=($!)
            if [ $(( (i + 1) % CONCURRENCY )) -eq 0 ] && [ $((i + 1)) -lt ${#RESEARCH_TASKS[@]} ]; then
                echo "Waiting for current Research batch to complete..."
                wait "${PIDS[@]}"
                PIDS=()
            fi
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        echo ">>> Research Tasks successfully completed!"
    fi
fi

# ------------------------------------------------------------------------------
# 3. Aggregate & Report Evaluation Results
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "📊 Aggregating Evaluation Results..."
echo "======================================================================"
$UV run python -m marble.experiments.evaluate \
    --run-dir "$OUT_DIR" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" || true

echo "======================================================================"
echo "✅ All $SPLIT episodes finished for baseline '$BASELINE'!"
echo "   Results directory: $OUT_DIR"
echo "======================================================================"
