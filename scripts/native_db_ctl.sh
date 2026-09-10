#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Native Non-Root Userland Database Controller (Path B)
# Runs PostgreSQL 16, Prometheus, node_exporter, and postgres_exporter
# Completely in userland without Docker or sudo privileges!
# ==============================================================================

ACTION="${1:-}"
WORKER_ID="${2:-0}"
DB_PORT="${3:-54320}"
PROM_PORT="${4:-55000}"
NODE_PORT="${5:-56000}"
PG_EXP_PORT="${6:-57000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect pg_env bin directory
if [ -d "/data/home/huangzixuan/miniconda3/envs/pg_env/bin" ]; then
    PG_ENV_BIN="/data/home/huangzixuan/miniconda3/envs/pg_env/bin"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/postgres" ]; then
    PG_ENV_BIN="$CONDA_PREFIX/bin"
else
    PG_ENV_BIN="$(dirname "$(command -v postgres 2>/dev/null || echo "/usr/bin/postgres")")"
fi
export PATH="$PG_ENV_BIN:$PATH"

RUNTIME_DIR="$REPO_ROOT/runs/native_db_w${WORKER_ID}"
PID_FILE="$RUNTIME_DIR/pids.env"

case "$ACTION" in
    up)
        # Ensure clean state for this worker slot
        if [ -f "$PID_FILE" ]; then
            source "$PID_FILE" || true
            kill "${PG_PID:-}" "${PROM_PID:-}" "${NODE_PID:-}" "${PG_EXP_PID:-}" 2>/dev/null || true
        fi
        rm -rf "$RUNTIME_DIR"
        mkdir -p "$RUNTIME_DIR/pgdata" "$RUNTIME_DIR/logs" "$RUNTIME_DIR/prom_data"

        # 1. Initialize PostgreSQL 16
        "$PG_ENV_BIN/initdb" -D "$RUNTIME_DIR/pgdata" -U test -A trust >/dev/null

        # 2. Start PostgreSQL 16
        "$PG_ENV_BIN/pg_ctl" -D "$RUNTIME_DIR/pgdata" \
            -o "-p $DB_PORT -c shared_preload_libraries=pg_stat_statements -c pg_stat_statements.track=all -c max_connections=300 -c log_min_error_statement=info -c logging_collector=on -c unix_socket_directories=$RUNTIME_DIR -h 127.0.0.1" \
            -l "$RUNTIME_DIR/logs/postgres.log" start >/dev/null

        # Wait for PostgreSQL to be ready
        for attempt in $(seq 1 20); do
            if "$PG_ENV_BIN/pg_isready" -h 127.0.0.1 -p "$DB_PORT" >/dev/null 2>&1; then
                break
            fi
            sleep 0.3
        done

        # 3. Create sysbench database & enable pg_stat_statements
        "$PG_ENV_BIN/createdb" -h 127.0.0.1 -p "$DB_PORT" -U test sysbench >/dev/null 2>&1 || true
        "$PG_ENV_BIN/psql" -h 127.0.0.1 -p "$DB_PORT" -U test -d sysbench \
            -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements; SELECT pg_stat_statements_reset();" >/dev/null 2>&1 || true

        # 4. Start node_exporter
        "$PG_ENV_BIN/node_exporter" \
            --web.listen-address="127.0.0.1:$NODE_PORT" \
            > "$RUNTIME_DIR/logs/node_exporter.log" 2>&1 &
        NODE_PID=$!

        # 5. Start postgres_exporter
        DATA_SOURCE_NAME="postgresql://test:Test123_456@127.0.0.1:${DB_PORT}/sysbench?sslmode=disable" \
        "$PG_ENV_BIN/postgres_exporter" \
            --web.listen-address="127.0.0.1:$PG_EXP_PORT" \
            > "$RUNTIME_DIR/logs/pg_exporter.log" 2>&1 &
        PG_EXP_PID=$!

        # 6. Generate Prometheus config with actual rule files
        cat << PROM_EOF > "$RUNTIME_DIR/prometheus.yml"
global:
  scrape_interval: 3s
  evaluation_interval: 60s
  scrape_timeout: 3s

rule_files:
  - "$REPO_ROOT/marble/environments/db_env_docker/node_rules.yml"
  - "$REPO_ROOT/marble/environments/db_env_docker/pgsql_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['127.0.0.1:$PROM_PORT']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['127.0.0.1:$NODE_PORT']

  - job_name: 'postgres_exporter'
    static_configs:
      - targets: ['127.0.0.1:$PG_EXP_PORT']
PROM_EOF

        # 7. Start Prometheus
        "$PG_ENV_BIN/prometheus" \
            --config.file="$RUNTIME_DIR/prometheus.yml" \
            --storage.tsdb.path="$RUNTIME_DIR/prom_data" \
            --web.listen-address="127.0.0.1:$PROM_PORT" \
            > "$RUNTIME_DIR/logs/prometheus.log" 2>&1 &
        PROM_PID=$!

        PG_PID="$(head -n 1 "$RUNTIME_DIR/pgdata/postmaster.pid" 2>/dev/null || echo "")"

        cat << PIDS_EOF > "$PID_FILE"
PG_PID=$PG_PID
NODE_PID=$NODE_PID
PG_EXP_PID=$PG_EXP_PID
PROM_PID=$PROM_PID
PIDS_EOF
        echo "[Native DB Worker $WORKER_ID] ✅ Up on DB:$DB_PORT Prom:$PROM_PORT Node:$NODE_PORT PgExp:$PG_EXP_PORT"
        ;;

    down)
        if [ -f "$PID_FILE" ]; then
            source "$PID_FILE" || true
            kill "${NODE_PID:-}" "${PG_EXP_PID:-}" "${PROM_PID:-}" 2>/dev/null || true
        fi
        if [ -d "$RUNTIME_DIR/pgdata" ]; then
            "$PG_ENV_BIN/pg_ctl" -D "$RUNTIME_DIR/pgdata" -m fast stop >/dev/null 2>&1 || true
        fi
        rm -rf "$RUNTIME_DIR"
        echo "[Native DB Worker $WORKER_ID] 🛑 Down and cleaned up."
        ;;

    cleanup_all)
        echo "Cleaning up all native DB worker runtimes..."
        for pfile in "$REPO_ROOT"/runs/native_db_w*/pids.env; do
            if [ -f "$pfile" ]; then
                source "$pfile" || true
                kill "${PG_PID:-}" "${NODE_PID:-}" "${PG_EXP_PID:-}" "${PROM_PID:-}" 2>/dev/null || true
            fi
        done
        for pgdir in "$REPO_ROOT"/runs/native_db_w*/pgdata; do
            if [ -d "$pgdir" ]; then
                "$PG_ENV_BIN/pg_ctl" -D "$pgdir" -m fast stop >/dev/null 2>&1 || true
            fi
        done
        # Kill any stray processes by binary name if still lingering
        pkill -f "postgres.*runs/native_db_w" 2>/dev/null || true
        pkill -f "prometheus.*runs/native_db_w" 2>/dev/null || true
        pkill -f "node_exporter.*runs/native_db_w" 2>/dev/null || true
        pkill -f "postgres_exporter.*runs/native_db_w" 2>/dev/null || true
        rm -rf "$REPO_ROOT"/runs/native_db_w*
        echo "✅ All native DB runtimes wiped clean."
        ;;

    *)
        echo "Usage: $0 {up|down|cleanup_all} [worker_id] [db_port] [prom_port] [node_port] [pg_exp_port]"
        exit 1
        ;;
esac
