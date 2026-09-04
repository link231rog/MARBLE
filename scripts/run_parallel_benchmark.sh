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
CONCURRENCY=8
ENABLE_COMM_GOV=""
DRY_RUN=""
OUT_DIR=""

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
        -c|--concurrency)
            CONCURRENCY="$2"
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
echo "  Baseline:   $BASELINE"
echo "  Split:      $SPLIT"
echo "  Benchmark:  $BENCHMARK_TARGET"
echo "  Max Cards:  $MAX_CARDS"
echo "  Output Dir: $OUT_DIR"
echo "======================================================================"

# Parse API keys for rotation if multiple keys are provided
IFS=',' read -r -a QWEN_KEYS <<< "${MARBLE_QWEN_API_KEYS:-${MARBLE_QWEN_API_KEY:-}}"
IFS=',' read -r -a WORKER_KEYS <<< "${MARBLE_WORKER_API_KEYS:-${NVAPI_KEY:-${OPENAI_API_KEY:-}}}"

get_qwen_key() {
    local idx=$1
    if [ ${#QWEN_KEYS[@]} -gt 0 ]; then
        echo "${QWEN_KEYS[$((idx % ${#QWEN_KEYS[@]}))]}"
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
# Execution Flow
# If CONCURRENCY >= 16 and BENCHMARK_TARGET == "both", launch all Database AND Research tasks simultaneously!
# Otherwise, run in phased batches.
# ------------------------------------------------------------------------------
if [ "$BENCHMARK_TARGET" == "both" ] && [ "$CONCURRENCY" -ge 16 ]; then
    echo "======================================================================"
    echo "🚀 [16-Concurrency Simultaneous Mode] Launching Database & Research in Parallel!"
    echo "  Database Workers: 0..$(( ${#DB_TASKS[@]} - 1 ))"
    echo "  Research Workers: ${#DB_TASKS[@]}..$(( ${#DB_TASKS[@]} + ${#RESEARCH_TASKS[@]} - 1 ))"
    echo "======================================================================"
    PIDS=()

    # Launch Database workers (0..7)
    for i in "${!DB_TASKS[@]}"; do
        task_id="${DB_TASKS[$i]}"
        worker_id=$i
        db_port=$((54320 + worker_id))
        prom_port=$((59090 + worker_id))
        node_port=$((59100 + worker_id))
        pg_exp_port=$((59180 + worker_id))
        compose_proj="marble_db_w${worker_id}"

        q_key="$(get_qwen_key "$worker_id")"
        w_key="$(get_worker_key "$worker_id")"

        (
            export MARBLE_DB_PORT="$db_port"
            export MARBLE_PROM_PORT="$prom_port"
            export MARBLE_NODE_PORT="$node_port"
            export MARBLE_PG_EXPORTER_PORT="$pg_exp_port"
            export MARBLE_COMPOSE_PROJECT="$compose_proj"
            [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
            [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

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
                    --max-iterations 5 \
                    --retrieval visible_k \
                    --max-cards "$MAX_CARDS" \
                    --lambda 0.15 \
                    --beta 0.25 \
                    --task-timeout 900 \
                    $ENABLE_COMM_GOV \
                    --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_task_${task_id}_w${worker_id}.log"

                # Cleanup worker sandbox
                docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true
            fi
            echo "[DB Worker $worker_id] Finished Database Task $task_id."
        ) &
        PIDS+=($!)
    done

    # Launch Research workers (8..15)
    for j in "${!RESEARCH_TASKS[@]}"; do
        task_id="${RESEARCH_TASKS[$j]}"
        worker_id=$(( ${#DB_TASKS[@]} + j ))

        q_key="$(get_qwen_key "$worker_id")"
        w_key="$(get_worker_key "$worker_id")"

        (
            [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
            [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

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
                    --max-iterations 5 \
                    --retrieval visible_k \
                    --max-cards "$MAX_CARDS" \
                    --lambda 0.15 \
                    --beta 0.25 \
                    --task-timeout 900 \
                    $ENABLE_COMM_GOV \
                    --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_task_${task_id}_w${worker_id}.log"
            fi
            echo "[Research Worker $worker_id] Finished Research Task $task_id."
        ) &
        PIDS+=($!)
    done

    echo ">>> All ${#PIDS[@]} workers active in parallel! Waiting for full wave completion..."
    wait "${PIDS[@]}"
    echo ">>> All Database and Research Tasks successfully completed in parallel!"

else
    # ------------------------------------------------------------------------------
    # 1. Run Database Tasks (Sandboxed Docker per worker)
    # ------------------------------------------------------------------------------
    if [ "$BENCHMARK_TARGET" == "both" ] || [ "$BENCHMARK_TARGET" == "database" ]; then
        echo ">>> [Phase 1/2] Launching Database Benchmark across $CONCURRENCY parallel sandboxes..."
        PIDS=()
        for i in "${!DB_TASKS[@]}"; do
            task_id="${DB_TASKS[$i]}"
            worker_id=$((i % CONCURRENCY))
            db_port=$((54320 + worker_id))
            prom_port=$((59090 + worker_id))
            node_port=$((59100 + worker_id))
            pg_exp_port=$((59180 + worker_id))
            compose_proj="marble_db_w${worker_id}"

            q_key="$(get_qwen_key "$worker_id")"
            w_key="$(get_worker_key "$worker_id")"

            (
                export MARBLE_DB_PORT="$db_port"
                export MARBLE_PROM_PORT="$prom_port"
                export MARBLE_NODE_PORT="$node_port"
                export MARBLE_PG_EXPORTER_PORT="$pg_exp_port"
                export MARBLE_COMPOSE_PROJECT="$compose_proj"
                [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
                [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

                echo "[Worker $worker_id] Launching Database Task $task_id (Port: $db_port)..."
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
                        --max-iterations 5 \
                        --retrieval visible_k \
                        --max-cards "$MAX_CARDS" \
                        --lambda 0.15 \
                        --beta 0.25 \
                        --task-timeout 900 \
                        $ENABLE_COMM_GOV \
                        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_task_${task_id}_w${worker_id}.log"

                    # Cleanup worker sandbox
                    docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true
                fi
                echo "[Worker $worker_id] Finished Database Task $task_id."
            ) &
            PIDS+=($!)

            # If batch reached concurrency limit, wait for batch before next slot
            if [ $(( (i + 1) % CONCURRENCY )) -eq 0 ] && [ $((i + 1)) -lt ${#DB_TASKS[@]} ]; then
                echo "Waiting for current Database batch to complete..."
                wait "${PIDS[@]}"
                PIDS=()
            fi
        done

        # Wait for remaining database workers
        if [ ${#PIDS[@]} -gt 0 ]; then
            wait "${PIDS[@]}"
        fi
        echo ">>> Database Tasks successfully completed!"
    fi

    # ------------------------------------------------------------------------------
    # 2. Run Research Tasks (Pure in-memory)
    # ------------------------------------------------------------------------------
    if [ "$BENCHMARK_TARGET" == "both" ] || [ "$BENCHMARK_TARGET" == "research" ]; then
        echo ">>> [Phase 2/2] Launching Research Benchmark across $CONCURRENCY parallel workers..."
        PIDS=()
        for i in "${!RESEARCH_TASKS[@]}"; do
            task_id="${RESEARCH_TASKS[$i]}"
            worker_id=$((i % CONCURRENCY))

            q_key="$(get_qwen_key "$worker_id")"
            w_key="$(get_worker_key "$worker_id")"

            (
                [ -n "$q_key" ] && export MARBLE_QWEN_API_KEY="$q_key"
                [ -n "$w_key" ] && export NVAPI_KEY="$w_key"

                echo "[Worker $worker_id] Launching Research Task $task_id..."
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
                        --max-iterations 5 \
                        --retrieval visible_k \
                        --max-cards "$MAX_CARDS" \
                        --lambda 0.15 \
                        --beta 0.25 \
                        --task-timeout 900 \
                        $ENABLE_COMM_GOV \
                        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/research_task_${task_id}_w${worker_id}.log"
                fi
                echo "[Worker $worker_id] Finished Research Task $task_id."
            ) &
            PIDS+=($!)

            if [ $(( (i + 1) % CONCURRENCY )) -eq 0 ] && [ $((i + 1)) -lt ${#RESEARCH_TASKS[@]} ]; then
                echo "Waiting for current Research batch to complete..."
                wait "${PIDS[@]}"
                PIDS=()
            fi
        done

        if [ ${#PIDS[@]} -gt 0 ]; then
            wait "${PIDS[@]}"
        fi
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
