#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Option B Master Hard Benchmark Runner (24 Test Hard / 12 Train Hard)
# Domains: Database, Research, Coding
# Dispatches MultiAgentBench tasks across sandboxed parallel workers.
# Automatically allocates unique ports and Docker compose projects for Database.
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

if [ -n "${PYTHON_BIN:-}" ]; then
    RUN_PY="$PYTHON_BIN"
elif [ -f "/data/home/huangzixuan/miniconda3/envs/sigir/bin/python" ]; then
    RUN_PY="/data/home/huangzixuan/miniconda3/envs/sigir/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    RUN_PY="$CONDA_PREFIX/bin/python"
elif command -v uv >/dev/null 2>&1 && [ -f "$REPO_DIR/.venv/bin/python" ]; then
    RUN_PY="uv run python"
elif [ -f "$REPO_DIR/.venv/bin/python" ]; then
    RUN_PY="$REPO_DIR/.venv/bin/python"
else
    RUN_PY="python3"
fi
MANIFEST="configs/experiments/multiagentbench_hard_frozen.json"

# Default arguments
BASELINE="ours_rl"
SPLIT="test_hard"
BENCHMARK_TARGET="all"
MAX_CARDS=5
MAX_ITERATIONS=5
CONCURRENCY="${CONCURRENCY:-16}"
ENABLE_COMM_GOV=""
DRY_RUN="${DRY_RUN:-}"
OUT_DIR=""
CONTROLLER_CHECKPOINT=""
LAMBDA="0.15"
BETA="0.25"
ABLATION=""
MARBLE_DB_RUNTIME="${MARBLE_DB_RUNTIME:-auto}"

PORT_OFFSET=0
SEED="42"
GPU1_ON_DEMAND="${MARBLE_GPU1_ON_DEMAND:-0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-0}"
QWEN_TEMP="${QWEN_TEMP:-}"

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-runtime)
            MARBLE_DB_RUNTIME="$2"
            shift 2
            ;;
        --task-timeout)
            TASK_TIMEOUT="$2"
            shift 2
            ;;
        --gpu1-on-demand)
            GPU1_ON_DEMAND="1"
            shift 1
            ;;
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
        --multi-out)
            MULTI_OUT="$2"
            shift 2
            ;;
        -o|--out)
            OUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --qwen-api-model)
            QWEN_API_MODEL="$2"
            export MARBLE_QWEN_API_MODELS="$2"
            shift 2
            ;;
        --qwen-api-base)
            QWEN_API_BASE="$2"
            export MARBLE_QWEN_API_BASES="$2"
            shift 2
            ;;
        --qwen-temperature)
            QWEN_TEMP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if baseline was pruned
if [ "$BASELINE" == "amem_style" ] || [ "$BASELINE" == "memoryos_style" ]; then
    echo "======================================================================"
    echo "⚡ Baseline $BASELINE has been pruned (Option B: Single-Agent Memory represented by canonical Mem0)."
    echo "⚡ Skipping execution."
    echo "======================================================================"
    exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [ -n "${MULTI_OUT:-}" ]; then
    IFS=',' read -r -a TARGET_OUT_DIRS <<< "$MULTI_OUT"
    OUT_DIR="${TARGET_OUT_DIRS[0]}"
elif [[ "$OUT_DIR" == *","* ]]; then
    IFS=',' read -r -a TARGET_OUT_DIRS <<< "$OUT_DIR"
    OUT_DIR="${TARGET_OUT_DIRS[0]}"
else
    if [ -z "$OUT_DIR" ]; then
        OUT_DIR="runs/hard_${BASELINE}_${SPLIT}_c${CONCURRENCY}_${TIMESTAMP}"
    fi
    TARGET_OUT_DIRS=("$OUT_DIR")
fi
mkdir -p "${TARGET_OUT_DIRS[@]}"

echo "======================================================================"
echo "⚡ MARBLE Master Hard Benchmark Runner (Option B: 24 Test / 12 Train Hard)"
echo "  Baseline:    $BASELINE"
echo "  Split:       $SPLIT"
echo "  Benchmark:   $BENCHMARK_TARGET"
echo "  Manifest:    $MANIFEST"
echo "  Max Cards:   $MAX_CARDS"
echo "  Concurrency: $CONCURRENCY"
echo "  Port Offset: $PORT_OFFSET"
if [ ${#TARGET_OUT_DIRS[@]} -gt 1 ]; then
    echo "  Multi-Out Targets (${#TARGET_OUT_DIRS[@]}): ${TARGET_OUT_DIRS[*]}"
else
    echo "  Output Dir:  $OUT_DIR"
fi
[ -n "$CONTROLLER_CHECKPOINT" ] && echo "  Checkpoint:  $CONTROLLER_CHECKPOINT"
[ -n "$ABLATION" ] && echo "  Ablation:    $ABLATION"

# Resolve Database runtime (native userland vs docker compose)
if [ "$MARBLE_DB_RUNTIME" == "native" ]; then
    DB_RUNTIME="native"
elif [ "$MARBLE_DB_RUNTIME" == "docker" ]; then
    DB_RUNTIME="docker"
elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    DB_RUNTIME="docker"
else
    DB_RUNTIME="native"
fi
echo "  DB Runtime:  $DB_RUNTIME"

# Robust vLLM endpoint resolution for 'ours_*' baselines
if [[ "$BASELINE" =~ ^ours_ ]]; then
    if [ -z "${QWEN_API_BASE:-}" ]; then
        if curl -s --connect-timeout 2 "http://127.0.0.1:8000/v1/health" >/dev/null 2>&1; then
            TARGET_BASE="http://127.0.0.1:8000/v1"
        else
            TARGET_BASE="http://127.0.0.1:18000/v1"
        fi
    else
        TARGET_BASE="$QWEN_API_BASE"
    fi

    echo "⚡ Checking vLLM health on $TARGET_BASE..."
    HEALTH_OK=false
    for attempt in 1 2 3 4 5; do
        if curl -s --connect-timeout 3 "$TARGET_BASE/health" >/dev/null 2>&1; then
            HEALTH_OK=true
            break
        fi
        sleep 1
    done

    if [ "$HEALTH_OK" = true ]; then
        echo "⚡ Detected active vLLM on $TARGET_BASE."
        export MARBLE_QWEN_API_KEYS="EMPTY"
        export MARBLE_QWEN_API_BASES="$TARGET_BASE"
        if [ -z "${QWEN_API_MODEL:-}" ]; then
            if [ "$BASELINE" == "ours_sft" ]; then
                export MARBLE_QWEN_API_MODELS="qwen_sft"
            elif [ "$BASELINE" == "ours_rl" ] || [ "$BASELINE" == "ours_private_to_global" ]; then
                export MARBLE_QWEN_API_MODELS="qwen_rl"
            else
                export MARBLE_QWEN_API_MODELS="Qwen/Qwen3.5-4B"
            fi
        else
            export MARBLE_QWEN_API_MODELS="$QWEN_API_MODEL"
        fi
    else
        echo "❌ [FATAL ERROR] Baseline '$BASELINE' requires controller endpoint at $TARGET_BASE, but health check failed!" >&2
        echo "   Please ensure vLLM is running (either directly on port 8000 or via SSH tunnel on port 18000)." >&2
        exit 1
    fi
fi

# Parse API keys and endpoints for multi-provider rotation
IFS=',' read -r -a QWEN_KEYS <<< "${MARBLE_QWEN_API_KEYS:-${MARBLE_QWEN_API_KEY:-}}"
IFS=',' read -r -a QWEN_BASES <<< "${MARBLE_QWEN_API_BASES:-${MARBLE_QWEN_API_BASE:-}}"
IFS=',' read -r -a QWEN_MODELS <<< "${MARBLE_QWEN_API_MODELS:-${MARBLE_QWEN_API_MODEL:-}}"
IFS=',' read -r -a WORKER_KEYS <<< "${MARBLE_WORKER_API_KEYS:-${NVIDIA_API_KEY:-${NVAPI_KEY:-${OPENAI_API_KEY:-}}}}"

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
read -r -a DB_TASKS <<< "$("$RUN_PY" -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='database'))" 2>/dev/null || echo "")"
read -r -a RESEARCH_TASKS <<< "$("$RUN_PY" -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='research'))" 2>/dev/null || echo "")"
read -r -a CODING_TASKS <<< "$("$RUN_PY" -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='coding'))" 2>/dev/null || echo "")"

echo "Loaded Tasks from Manifest ($MANIFEST, split: $SPLIT):"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "database" ] && echo "  Database Tasks (${#DB_TASKS[@]}): ${DB_TASKS[*]:-}"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "research" ] && echo "  Research Tasks (${#RESEARCH_TASKS[@]}): ${RESEARCH_TASKS[*]:-}"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "coding" ] && echo "  Coding Tasks   (${#CODING_TASKS[@]}): ${CODING_TASKS[*]:-}"

EXTRA_ARGS=""
[ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
[ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
[ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"

# ------------------------------------------------------------------------------
# Task Dispatcher Helpers
# ------------------------------------------------------------------------------
launch_db_worker() {
    local task_id="$1"
    local worker_id="$2"
    local target_out="${3:-$OUT_DIR}"
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
        export PGPORT="$db_port"
        export PROMETHEUS_PORT="$prom_port"
        export NODE_EXPORTER_PORT="$node_port"
        export POSTGRES_EXPORTER_PORT="$pg_exp_port"

        if [ "$DB_RUNTIME" == "native" ]; then
            trap '"$REPO_DIR/scripts/native_db_ctl.sh" down "$worker_id" >/dev/null 2>&1 || true' EXIT INT TERM
        else
            trap 'docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true' EXIT INT TERM
        fi

        export MARBLE_DB_RUNTIME="$DB_RUNTIME"
        export MARBLE_MANAGED_STACK="1"
        export MARBLE_WORKER_ID="$worker_id"
        export MARBLE_DB_PORT="$db_port"
        export MARBLE_PROM_PORT="$prom_port"
        export MARBLE_NODE_PORT="$node_port"
        export MARBLE_PG_EXPORTER_PORT="$pg_exp_port"
        export MARBLE_COMPOSE_PROJECT="$compose_proj"

        [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
        [ -n "$q_base" ] && export MARBLE_QWEN_API_BASE="$q_base"
        [ -n "$q_model" ] && export MARBLE_QWEN_API_MODEL="$q_model"
        [ -n "$w_key" ] && { export NVAPI_KEY="$w_key"; export NVIDIA_API_KEY="$w_key"; export OPENAI_API_KEY="$w_key"; }

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"
        [ -n "$q_base" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-base $q_base"
        [ -n "$q_model" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-model $q_model"
        [ -n "$q_key" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-key $q_key"
        [ -n "$QWEN_TEMP" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-temperature $QWEN_TEMP"

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[DB Worker $worker_id] Launching DB Task $task_id ($DB_RUNTIME mode, DB:$db_port, Prom:$prom_port)..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $RUN_PY -u -m marble.experiments.run_benchmark --benchmark database --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if [ "$DB_RUNTIME" == "native" ]; then
                "$REPO_DIR/scripts/native_db_ctl.sh" up "$worker_id" "$db_port" "$prom_port" "$node_port" "$pg_exp_port" >/dev/null 2>&1 || true
            else
                if ! docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml up -d >/dev/null 2>&1; then
                    sleep 4
                    docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml up -d >/dev/null 2>&1 || true
                fi
            fi
            sleep 2

            if ! $RUN_PY -u -m marble.experiments.run_benchmark \
                --benchmark database \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --task-ids "$task_id" \
                --baseline "$BASELINE" \
                --seed "$SEED" \
                --max-iterations "$MAX_ITERATIONS" \
                --retrieval visible_k \
                --max-cards "$MAX_CARDS" \
                --lambda "$LAMBDA" \
                --beta "$BETA" \
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$target_out" 2>&1 | tee -a "$target_out/database_task_${task_id}_w${worker_id}.log"; then
                echo "[DB Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
                $RUN_PY -u -m marble.experiments.run_benchmark \
                    --benchmark database \
                    --manifest "$MANIFEST" \
                    --split "$SPLIT" \
                    --task-ids "$task_id" \
                    --baseline "$BASELINE" \
                    --seed "$SEED" \
                    --max-iterations "$MAX_ITERATIONS" \
                    --retrieval visible_k \
                    --max-cards "$MAX_CARDS" \
                    --lambda "$LAMBDA" \
                    --beta "$BETA" \
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$target_out" 2>&1 | tee -a "$target_out/database_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[DB Worker $worker_id] Finished DB Task $task_id."
    )
}

launch_research_worker() {
    local task_id="$1"
    local worker_id="$2"
    local target_out="${3:-$OUT_DIR}"
    local q_key="$(get_qwen_key "$worker_id")"
    local q_base="$(get_qwen_base "$worker_id")"
    local q_model="$(get_qwen_model "$worker_id")"
    local w_key="$(get_worker_key "$worker_id")"

    (
        [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
        [ -n "$q_base" ] && export MARBLE_QWEN_API_BASE="$q_base"
        [ -n "$q_model" ] && export MARBLE_QWEN_API_MODEL="$q_model"
        [ -n "$w_key" ] && { export NVAPI_KEY="$w_key"; export NVIDIA_API_KEY="$w_key"; export OPENAI_API_KEY="$w_key"; }

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"
        [ -n "$q_base" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-base $q_base"
        [ -n "$q_model" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-model $q_model"
        [ -n "$q_key" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-key $q_key"
        [ -n "$QWEN_TEMP" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-temperature $QWEN_TEMP"

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[Research Worker $worker_id] Launching Research Task $task_id..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $RUN_PY -u -m marble.experiments.run_benchmark --benchmark research --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if ! $RUN_PY -u -m marble.experiments.run_benchmark \
                --benchmark research \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --task-ids "$task_id" \
                --baseline "$BASELINE" \
                --seed "$SEED" \
                --max-iterations "$MAX_ITERATIONS" \
                --retrieval visible_k \
                --max-cards "$MAX_CARDS" \
                --lambda "$LAMBDA" \
                --beta "$BETA" \
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$target_out" 2>&1 | tee -a "$target_out/research_task_${task_id}_w${worker_id}.log"; then
                echo "[Research Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
                $RUN_PY -u -m marble.experiments.run_benchmark \
                    --benchmark research \
                    --manifest "$MANIFEST" \
                    --split "$SPLIT" \
                    --task-ids "$task_id" \
                    --baseline "$BASELINE" \
                    --seed "$SEED" \
                    --max-iterations "$MAX_ITERATIONS" \
                    --retrieval visible_k \
                    --max-cards "$MAX_CARDS" \
                    --lambda "$LAMBDA" \
                    --beta "$BETA" \
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$target_out" 2>&1 | tee -a "$target_out/research_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[Research Worker $worker_id] Finished Research Task $task_id."
    )
}

launch_coding_worker() {
    local task_id="$1"
    local worker_id="$2"
    local target_out="${3:-$OUT_DIR}"
    local q_key="$(get_qwen_key "$worker_id")"
    local q_base="$(get_qwen_base "$worker_id")"
    local q_model="$(get_qwen_model "$worker_id")"
    local w_key="$(get_worker_key "$worker_id")"

    (
        [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
        [ -n "$q_base" ] && export MARBLE_QWEN_API_BASE="$q_base"
        [ -n "$q_model" ] && export MARBLE_QWEN_API_MODEL="$q_model"
        [ -n "$w_key" ] && { export NVAPI_KEY="$w_key"; export NVIDIA_API_KEY="$w_key"; export OPENAI_API_KEY="$w_key"; }

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"
        [ -n "$q_base" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-base $q_base"
        [ -n "$q_model" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-model $q_model"
        [ -n "$q_key" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-api-key $q_key"
        [ -n "$QWEN_TEMP" ] && EXTRA_ARGS="$EXTRA_ARGS --qwen-temperature $QWEN_TEMP"

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[Coding Worker $worker_id] Launching Coding Task $task_id..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $RUN_PY -u -m marble.experiments.run_benchmark --benchmark coding --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if ! $RUN_PY -u -m marble.experiments.run_benchmark \
                --benchmark coding \
                --manifest "$MANIFEST" \
                --split "$SPLIT" \
                --task-ids "$task_id" \
                --baseline "$BASELINE" \
                --seed "$SEED" \
                --max-iterations "$MAX_ITERATIONS" \
                --retrieval visible_k \
                --max-cards "$MAX_CARDS" \
                --lambda "$LAMBDA" \
                --beta "$BETA" \
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$target_out" 2>&1 | tee -a "$target_out/coding_task_${task_id}_w${worker_id}.log"; then
                echo "[Coding Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
                $RUN_PY -u -m marble.experiments.run_benchmark \
                    --benchmark coding \
                    --manifest "$MANIFEST" \
                    --split "$SPLIT" \
                    --task-ids "$task_id" \
                    --baseline "$BASELINE" \
                    --seed "$SEED" \
                    --max-iterations "$MAX_ITERATIONS" \
                    --retrieval visible_k \
                    --max-cards "$MAX_CARDS" \
                    --lambda "$LAMBDA" \
                    --beta "$BETA" \
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$target_out" 2>&1 | tee -a "$target_out/coding_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[Coding Worker $worker_id] Finished Coding Task $task_id."
    )
}

# ------------------------------------------------------------------------------
# Execution Flow
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "🚀 [Dynamic Worker Pool Mode] Launching across $CONCURRENCY parallel worker slots!"
echo "======================================================================"

QUEUE_FILE="$OUT_DIR/.task_queue"
mkdir -p "${TARGET_OUT_DIRS[@]}"
> "$QUEUE_FILE"

# Enqueue all tasks across all target output directories
for target_out in "${TARGET_OUT_DIRS[@]}"; do
    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "database" ]; then
        for tid in ${DB_TASKS[@]+"${DB_TASKS[@]}"}; do
            [ -n "$tid" ] && echo "$target_out:database:$tid" >> "$QUEUE_FILE"
        done
    fi
    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "research" ]; then
        for tid in ${RESEARCH_TASKS[@]+"${RESEARCH_TASKS[@]}"}; do
            [ -n "$tid" ] && echo "$target_out:research:$tid" >> "$QUEUE_FILE"
        done
    fi
    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "coding" ]; then
        for tid in ${CODING_TASKS[@]+"${CODING_TASKS[@]}"}; do
            [ -n "$tid" ] && echo "$target_out:coding:$tid" >> "$QUEUE_FILE"
        done
    fi
done

TOTAL_QUEUED=$(wc -l < "$QUEUE_FILE" | tr -d " ")
echo "Loaded $TOTAL_QUEUED total tasks across ${#TARGET_OUT_DIRS[@]} target directories into Dynamic Worker Pool queue."

pop_dynamic_task() {
    "$RUN_PY" -c "
import fcntl, sys
qfile = '$QUEUE_FILE'
try:
    with open(qfile, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = f.readlines()
        if lines:
            task = lines[0].strip()
            f.seek(0)
            f.truncate()
            f.writelines(lines[1:])
            print(task)
        fcntl.flock(f, fcntl.LOCK_UN)
except Exception:
    pass
"
}

worker_slot_loop() {
    local slot_id="$1"
    echo "[Worker Pool Slot $slot_id] Online and listening for tasks..."
    while true; do
        local item="$(pop_dynamic_task)"
        if [ -z "$item" ]; then
            echo "[Worker Pool Slot $slot_id] No remaining tasks in queue. Slot exiting."
            break
        fi

        local target_out="$OUT_DIR"
        local bench=""
        local tid=""

        # Support multi-target item format: <target_out_dir>:<benchmark>:<task_id>
        local f1="" f2="" f3=""
        IFS=':' read -r f1 f2 f3 <<< "$item"
        if [ -n "$f3" ]; then
            target_out="$f1"
            bench="$f2"
            tid="$f3"
        else
            target_out="$OUT_DIR"
            bench="$f1"
            tid="$f2"
        fi

        mkdir -p "$target_out"

        # Fast-path idempotency: skip if already completed with status: ok
        local sum_file="$(find "$target_out" -path "*/$bench/$tid/summary.json" 2>/dev/null | head -n 1)"
        if [ -n "$sum_file" ] && [ -f "$sum_file" ]; then
            if grep -q '"status": "ok"' "$sum_file" 2>/dev/null; then
                echo "[Worker Pool Slot $slot_id] ⏭️ Task $bench $tid in $(basename "$target_out") already completed (status=ok). Skipping."
                continue
            fi
        fi

        echo "[Worker Pool Slot $slot_id] ==> Claimed task: $bench $tid (Target: $(basename "$target_out"))"
        if [ "$bench" == "database" ]; then
            launch_db_worker "$tid" "$slot_id" "$target_out"
        elif [ "$bench" == "research" ]; then
            launch_research_worker "$tid" "$slot_id" "$target_out"
        elif [ "$bench" == "coding" ]; then
            launch_coding_worker "$tid" "$slot_id" "$target_out"
        fi
        echo "[Worker Pool Slot $slot_id] <== Completed task: $bench $tid (Target: $(basename "$target_out"))"
    done
}

PIDS=()
for slot in $(seq 0 $((CONCURRENCY - 1))); do
    worker_slot_loop "$slot" &
    PIDS+=($!)
done

echo ">>> All $CONCURRENCY dynamic worker slots dispatched! Processing queue..."
wait "${PIDS[@]}"
echo ">>> All dynamic worker slots successfully finished all tasks!"
# ------------------------------------------------------------------------------
# Automatic Self-Healing & 429/Failure Retry Sweep
# Guarantees that any transient rate-limit (429) or port-conflict failures are
# automatically retried and resolved before aggregation and exit.
# ------------------------------------------------------------------------------
check_and_retry_missing_tasks() {
    local target_dir="${1:-$OUT_DIR}"
    local max_retries=2
    local attempt=1

    while [ $attempt -le $max_retries ]; do
        local missing_json
        missing_json="$("$RUN_PY" -c "
import glob, json, sys

manifest_path = '$MANIFEST'
split = '$SPLIT'
target = '$BENCHMARK_TARGET'
out_dir = '$target_dir'

with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

expected = []
for t in manifest['splits'][split]:
    bn = str(t['benchmark'])
    if target == 'all' or target == bn:
        expected.append((bn, int(t['task_id'])))

summaries = glob.glob(f'{out_dir}/**/summary.json', recursive=True)
found = set()
for s in summaries:
    try:
        with open(s) as sf:
            d = json.load(sf)
            if d.get('status') == 'ok':
                found.add((str(d['benchmark']), int(d['task_id'])))
    except:
        pass

missing = [t for t in expected if t not in found]
print(json.dumps(missing))
" 2>/dev/null || echo "[]")"

        if [ "$missing_json" == "[]" ] || [ -z "$missing_json" ]; then
            echo "✅ All expected tasks verified present and completed (status=ok) in $target_dir!"
            break
        fi

        echo "======================================================================"
        echo "⚠️ [Auto-Healing Sweep $attempt/$max_retries] Detected missing or incomplete tasks in $target_dir:"
        echo "   Missing: $missing_json"
        echo "   Cleaning sandboxes and retrying failed/rate-limited tasks..."
        echo "======================================================================"

        if [ "$DB_RUNTIME" == "native" ]; then
            "$REPO_DIR/scripts/native_db_ctl.sh" cleanup_all >/dev/null 2>&1 || true
        else
            [ -n "$(docker ps -q 2>/dev/null)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
            docker volume prune -f >/dev/null 2>&1 || true
        fi
        sleep 4

        local db_w=0
        local res_w=8
        local cod_w=16

        while read -r bn tid; do
            [ -z "$bn" ] && continue
            echo "⚡ [Auto-Retry Parallel] Dispatching Baseline $BASELINE | $bn Task $tid (Target: $(basename "$target_dir"))..."
            if [ "$bn" == "database" ]; then
                launch_db_worker "$tid" "$db_w" "$target_dir" &
                db_w=$(( (db_w + 1) % 8 ))
            elif [ "$bn" == "research" ]; then
                launch_research_worker "$tid" "$res_w" "$target_dir" &
                res_w=$(( res_w + 1 ))
            elif [ "$bn" == "coding" ]; then
                launch_coding_worker "$tid" "$cod_w" "$target_dir" &
                cod_w=$(( cod_w + 1 ))
            fi
        done < <("$RUN_PY" -c "import json, sys; [print(f'{b} {t}') for b, t in json.loads('$missing_json')]")

        echo "⏳ Waiting for parallel retry tasks to complete..."
        wait || true
        echo "✅ Parallel retry sweep round complete."

        attempt=$((attempt + 1))
    done
}

for target_dir in "${TARGET_OUT_DIRS[@]}"; do
    check_and_retry_missing_tasks "$target_dir"
done

# ------------------------------------------------------------------------------
# Aggregate & Report Evaluation Results
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "📊 Aggregating Evaluation Results..."
echo "======================================================================"
for target_dir in "${TARGET_OUT_DIRS[@]}"; do
    $RUN_PY -m marble.experiments.evaluate \
        --run-dir "$target_dir" \
        --manifest "$MANIFEST" \
        --split "$SPLIT" || true
done

echo "======================================================================"
echo "✅ All $SPLIT episodes finished for baseline '$BASELINE'!"
echo "   Results directories: ${TARGET_OUT_DIRS[*]}"
echo "======================================================================"
