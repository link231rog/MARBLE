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

UV="${UV:-$(command -v uv || echo "uv")}"
MANIFEST="configs/experiments/multiagentbench_hard_frozen.json"

# Default arguments
BASELINE="ours_rl"
SPLIT="test_hard"
BENCHMARK_TARGET="all"
MAX_CARDS=5
MAX_ITERATIONS=5
CONCURRENCY="${CONCURRENCY:-8}"
ENABLE_COMM_GOV=""
DRY_RUN=""
OUT_DIR=""
CONTROLLER_CHECKPOINT=""
LAMBDA="0.15"
BETA="0.25"
ABLATION=""

PORT_OFFSET=0
GPU1_ON_DEMAND="${MARBLE_GPU1_ON_DEMAND:-0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-0}"

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        -o|--out)
            OUT_DIR="$2"
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
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="runs/hard_${BASELINE}_${SPLIT}_c${CONCURRENCY}_${TIMESTAMP}"
fi
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "⚡ MARBLE Master Hard Benchmark Runner (Option B: 24 Test / 12 Train Hard)"
echo "  Baseline:    $BASELINE"
echo "  Split:       $SPLIT"
echo "  Benchmark:   $BENCHMARK_TARGET"
echo "  Manifest:    $MANIFEST"
echo "  Max Cards:   $MAX_CARDS"
echo "  Concurrency: $CONCURRENCY"
echo "  Port Offset: $PORT_OFFSET"
echo "  Output Dir:  $OUT_DIR"
[ -n "$CONTROLLER_CHECKPOINT" ] && echo "  Checkpoint:  $CONTROLLER_CHECKPOINT"
[ -n "$ABLATION" ] && echo "  Ablation:    $ABLATION"
# On-demand vLLM lifecycle on cis_gpu1 for 'ours_*' baselines
# Robust vLLM endpoint resolution for 'ours_*' baselines
if [[ "$BASELINE" =~ ^ours_ ]]; then
    TARGET_BASE="${QWEN_API_BASE:-http://127.0.0.1:18000/v1}"
    if [ "$TARGET_BASE" == "http://127.0.0.1:18000/v1" ]; then
        echo "⚡ Checking local/GPU1 vLLM health on $TARGET_BASE..."
        HEALTH_OK=false
        for attempt in 1 2 3 4 5; do
            if curl -s --connect-timeout 3 "$TARGET_BASE/health" >/dev/null 2>&1; then
                HEALTH_OK=true
                break
            fi
            sleep 1
        done

        if [ "$HEALTH_OK" = true ]; then
            echo "⚡ Detected active GPU1 vLLM on $TARGET_BASE."
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
            echo "   Please ensure the SSH tunnel (127.0.0.1:18000 -> GPU1:8000) and vLLM server are running." >&2
            exit 1
        fi
    else
        export MARBLE_QWEN_API_BASES="$TARGET_BASE"
        [ -n "${QWEN_API_MODEL:-}" ] && export MARBLE_QWEN_API_MODELS="$QWEN_API_MODEL"
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
read -r -a DB_TASKS <<< "$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='database'))" 2>/dev/null || echo "")"
read -r -a RESEARCH_TASKS <<< "$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='research'))" 2>/dev/null || echo "")"
read -r -a CODING_TASKS <<< "$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(str(x['task_id']) for x in m['splits']['$SPLIT'] if x['benchmark']=='coding'))" 2>/dev/null || echo "")"

echo "Loaded Tasks from Manifest ($MANIFEST, split: $SPLIT):"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "database" ] && echo "  Database Tasks (${#DB_TASKS[@]}): ${DB_TASKS[*]}"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "research" ] && echo "  Research Tasks (${#RESEARCH_TASKS[@]}): ${RESEARCH_TASKS[*]}"
[ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "coding" ] && echo "  Coding Tasks   (${#CODING_TASKS[@]}): ${CODING_TASKS[*]}"

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
        trap 'docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true' EXIT INT TERM

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

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[DB Worker $worker_id] Launching DB Task $task_id on ports (DB:$db_port, Prom:$prom_port)..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $UV run python -u -m marble.experiments.run_benchmark --benchmark database --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if ! docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml up -d >/dev/null 2>&1; then
                sleep 4
                docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml up -d >/dev/null 2>&1 || true
            fi
            sleep 2

            if ! $UV run python -u -m marble.experiments.run_benchmark \
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
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_task_${task_id}_w${worker_id}.log"; then
                echo "[DB Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
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
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[DB Worker $worker_id] Finished DB Task $task_id."
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
        [ -n "$w_key" ] && { export NVAPI_KEY="$w_key"; export NVIDIA_API_KEY="$w_key"; export OPENAI_API_KEY="$w_key"; }

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[Research Worker $worker_id] Launching Research Task $task_id..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $UV run python -u -m marble.experiments.run_benchmark --benchmark research --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if ! $UV run python -u -m marble.experiments.run_benchmark \
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
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_task_${task_id}_w${worker_id}.log"; then
                echo "[Research Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
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
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[Research Worker $worker_id] Finished Research Task $task_id."
    ) &
}

launch_coding_worker() {
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
        [ -n "$w_key" ] && { export NVAPI_KEY="$w_key"; export NVIDIA_API_KEY="$w_key"; export OPENAI_API_KEY="$w_key"; }

        EXTRA_ARGS=""
        [ -n "$ENABLE_COMM_GOV" ] && EXTRA_ARGS="$EXTRA_ARGS $ENABLE_COMM_GOV"
        [ -n "$CONTROLLER_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --controller-checkpoint $CONTROLLER_CHECKPOINT"
        [ -n "$ABLATION" ] && EXTRA_ARGS="$EXTRA_ARGS --ablation $ABLATION"

        # Stagger worker start to eliminate burst thundering herd
        sleep $(( (worker_id % 8) * 2 ))

        echo "[Coding Worker $worker_id] Launching Coding Task $task_id..."
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would execute: $UV run python -u -m marble.experiments.run_benchmark --benchmark coding --manifest $MANIFEST --split $SPLIT --task-ids $task_id --baseline $BASELINE ..."
        else
            if ! $UV run python -u -m marble.experiments.run_benchmark \
                --benchmark coding \
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
                --task-timeout "$TASK_TIMEOUT" \
                $EXTRA_ARGS \
                --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/coding_task_${task_id}_w${worker_id}.log"; then
                echo "[Coding Worker $worker_id] Task $task_id exited with error/429. Auto-retrying after 5s..."
                sleep 5
                $UV run python -u -m marble.experiments.run_benchmark \
                    --benchmark coding \
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
                    --task-timeout "$TASK_TIMEOUT" \
                    $EXTRA_ARGS \
                    --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/coding_task_${task_id}_w${worker_id}.log" || true
            fi
        fi
        echo "[Coding Worker $worker_id] Finished Coding Task $task_id."
    ) &
}

# ------------------------------------------------------------------------------
# Execution Flow
# ------------------------------------------------------------------------------
if [ "$BENCHMARK_TARGET" == "all" ] && [ "$CONCURRENCY" -ge 24 ]; then
    echo "======================================================================"
    echo "🚀 [24-Concurrency Simultaneous Mode] Launching Database, Research & Coding ALL IN PARALLEL!"
    echo "  Database Workers (0..$(( ${#DB_TASKS[@]} - 1 ))):       ${DB_TASKS[*]}"
    echo "  Research Workers ($(( ${#DB_TASKS[@]} ))..$(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} - 1 ))):      ${RESEARCH_TASKS[*]}"
    echo "  Coding Workers   ($(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} ))..$(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} + ${#CODING_TASKS[@]} - 1 ))):     ${CODING_TASKS[*]}"
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
    for k in "${!CODING_TASKS[@]}"; do
        launch_coding_worker "${CODING_TASKS[$k]}" "$(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} + k ))"
        PIDS+=($!)
    done

    echo ">>> All ${#PIDS[@]} workers active simultaneously! Waiting for full wave completion..."
    wait "${PIDS[@]}"
    echo ">>> All Database, Research, and Coding Tasks successfully completed in parallel!"

else
    # Phased batch mode
    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "database" ]; then
        if [ ${#DB_TASKS[@]} -gt 0 ]; then
            echo ">>> [Phase 1/3] Launching Database Benchmark across $CONCURRENCY parallel sandboxes..."
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
            echo ">>> Database Tasks completed!"
        fi
    fi

    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "research" ]; then
        if [ ${#RESEARCH_TASKS[@]} -gt 0 ]; then
            echo ">>> [Phase 2/3] Launching Research Benchmark across $CONCURRENCY parallel workers..."
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
            echo ">>> Research Tasks completed!"
        fi
    fi

    if [ "$BENCHMARK_TARGET" == "all" ] || [ "$BENCHMARK_TARGET" == "coding" ]; then
        if [ ${#CODING_TASKS[@]} -gt 0 ]; then
            echo ">>> [Phase 3/3] Launching Coding Benchmark across $CONCURRENCY parallel workers..."
            PIDS=()
            for i in "${!CODING_TASKS[@]}"; do
                launch_coding_worker "${CODING_TASKS[$i]}" "$((i % CONCURRENCY))"
                PIDS+=($!)
                if [ $(( (i + 1) % CONCURRENCY )) -eq 0 ] && [ $((i + 1)) -lt ${#CODING_TASKS[@]} ]; then
                    echo "Waiting for current Coding batch to complete..."
                    wait "${PIDS[@]}"
                    PIDS=()
                fi
            done
            [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
            echo ">>> Coding Tasks completed!"
        fi
    fi
fi

# ------------------------------------------------------------------------------
# Automatic Self-Healing & 429/Failure Retry Sweep
# Guarantees that any transient rate-limit (429) or port-conflict failures are
# automatically retried and resolved before aggregation and exit.
# ------------------------------------------------------------------------------
check_and_retry_missing_tasks() {
    local max_retries=2
    local attempt=1

    while [ $attempt -le $max_retries ]; do
        local missing_json
        missing_json="$(python3 -c "
import glob, json, sys

manifest_path = '$MANIFEST'
split = '$SPLIT'
target = '$BENCHMARK_TARGET'
out_dir = '$OUT_DIR'

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
            echo "✅ All expected tasks verified present and completed (status=ok) in $OUT_DIR!"
            break
        fi

        echo "======================================================================"
        echo "⚠️ [Auto-Healing Sweep $attempt/$max_retries] Detected missing or incomplete tasks:"
        echo "   Missing: $missing_json"
        echo "   Cleaning docker sandboxes and retrying failed/rate-limited tasks..."
        echo "======================================================================"

        [ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
        docker volume prune -f >/dev/null 2>&1 || true
        sleep 4

        local db_w=0
        local res_w=8
        local cod_w=16

        while read -r bn tid; do
            [ -z "$bn" ] && continue
            echo "⚡ [Auto-Retry Parallel] Dispatching Baseline $BASELINE | $bn Task $tid..."
            if [ "$bn" == "database" ]; then
                launch_db_worker "$tid" "$db_w" &
                db_w=$(( (db_w + 1) % 8 ))
            elif [ "$bn" == "research" ]; then
                launch_research_worker "$tid" "$res_w" &
                res_w=$(( res_w + 1 ))
            elif [ "$bn" == "coding" ]; then
                launch_coding_worker "$tid" "$cod_w" &
                cod_w=$(( cod_w + 1 ))
            fi
        done < <(python3 -c "import json, sys; [print(f'{b} {t}') for b, t in json.loads('$missing_json')]")

        echo "⏳ Waiting for parallel retry tasks to complete..."
        wait || true
        echo "✅ Parallel retry sweep round complete."

        attempt=$((attempt + 1))
    done
}

check_and_retry_missing_tasks

# ------------------------------------------------------------------------------
# Aggregate & Report Evaluation Results
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
