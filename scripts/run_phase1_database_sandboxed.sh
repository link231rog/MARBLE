#!/bin/bash
set -u

# ==============================================================================
# MARBLE Sandboxed Parallel Database Benchmark Runner
# Spins up isolated Docker Compose projects with unique host port mappings
# allowing multiple Database benchmark workers to run concurrently on a single machine.
# ==============================================================================

unset NO_PROXY no_proxy
REPO_DIR="/Users/huangzixuan/Documents/404 not found/Good Night/Research/MAS&Memory/code/MARBLE"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
OUT_DIR="runs/nvidia-gpt-oss-stratified-20260904"
mkdir -p "$OUT_DIR"
MANIFEST="configs/experiments/multiagentbench_stratified_frozen.json"

# Helper function to run a slice of database tasks in an isolated Docker sandbox
run_sandboxed_database_worker() {
    local worker_id="$1"
    local baseline="$2"
    local split="$3"
    local task_ids="$4"

    # Dynamic port allocations per worker
    local db_port=$((54320 + worker_id))
    local prom_port=$((59090 + worker_id))
    local node_port=$((59100 + worker_id))
    local pg_exp_port=$((59180 + worker_id))
    local compose_proj="marble_db_w${worker_id}"

    echo "========================================================"
    echo "[DB Worker $worker_id] Starting Sandboxed Run"
    echo "  Baseline: $baseline | Split: $split | Tasks: $task_ids"
    echo "  Project:  $compose_proj"
    echo "  Ports:    DB=$db_port | Prom=$prom_port | Node=$node_port | PGExp=$pg_exp_port"
    echo "========================================================"

    # Export sandbox parameters into worker subshell environment
    export MARBLE_DB_PORT="$db_port"
    export MARBLE_PROM_PORT="$prom_port"
    export MARBLE_NODE_PORT="$node_port"
    export MARBLE_PG_EXPORTER_PORT="$pg_exp_port"
    export MARBLE_COMPOSE_PROJECT="$compose_proj"

    $UV run python -u -m marble.experiments.run_benchmark \
        --benchmark database \
        --manifest "$MANIFEST" \
        --split "$split" \
        --task-ids "$task_ids" \
        --baseline "$baseline" \
        --seed 42 \
        --max-iterations 5 \
        --retrieval visible_k \
        --max-cards 5 \
        --lambda 0.15 \
        --beta 0.25 \
        --task-timeout 900 \
        --out "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/database_${split}_${baseline}_w${worker_id}.log"

    # Cleanup worker-specific containers and volumes
    docker compose -p "$compose_proj" -f marble/environments/db_env_docker/docker-compose.yml down -v >/dev/null 2>&1 || true
    echo "[DB Worker $worker_id] Cleanup complete."
}

export -f run_sandboxed_database_worker

chmod +x scripts/run_phase1_database_sandboxed.sh
