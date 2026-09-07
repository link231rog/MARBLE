#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Hard Benchmark: Missing Task Backfill Utility
# Scans completed/partial baseline directories for any missing episodes
# out of the 24 frozen test_hard tasks, and runs only the missing tasks.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

MANIFEST="${1:-configs/experiments/multiagentbench_hard_frozen.json}"
SPLIT="${2:-test_hard}"

echo "======================================================================"
echo "🔍 MARBLE Benchmark: Checking for missing tasks..."
echo "  Manifest: $MANIFEST"
echo "  Split:    $SPLIT"
echo "======================================================================"

python3 - "$MANIFEST" "$SPLIT" << 'EOF'
import glob, json, os, subprocess, sys

manifest_path = sys.argv[1]
split = sys.argv[2]

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

expected_tasks = [(str(t["benchmark"]), int(t["task_id"])) for t in manifest["splits"][split]]
total_expected = len(expected_tasks)

run_dirs = sorted(glob.glob(f"runs/hard_*_{split}_c24_*"))
if not run_dirs:
    print("No baseline runs found matching runs/hard_*_${split}_c24_*")
    sys.exit(0)

for rd in run_dirs:
    summaries = glob.glob(f"{rd}/**/summary.json", recursive=True)
    b_name = None
    found_tasks = set()
    for s in summaries:
        try:
            with open(s) as sf:
                d = json.load(sf)
                found_tasks.add((str(d["benchmark"]), int(d["task_id"])))
                if not b_name and d.get("method"):
                    b_name = d.get("method")
        except:
            pass

    if not b_name:
        continue

    missing = [t for t in expected_tasks if t not in found_tasks]
    if not missing:
        print(f"✅ Baseline '{b_name}' in {rd}: 24/24 Complete.")
        continue

    print(f"⚠️ Baseline '{b_name}' in {rd}: {len(found_tasks)}/{total_expected} tasks. Missing: {missing}")
    uv = os.environ.get("UV", "uv")

    for benchmark, task_id in missing:
        print(f"⚡ Backfilling {b_name} | {benchmark} Task {task_id} into {rd}...")
        compose_proj = f"marble_db_{b_name}_backfill_{task_id}"

        if benchmark == "database":
            db_port = 54324
            prom_port = 55004
            node_port = 56004
            pg_exp_port = 57004
            env = os.environ.copy()
            env["MARBLE_DB_PORT"] = str(db_port)
            env["MARBLE_PROM_PORT"] = str(prom_port)
            env["MARBLE_NODE_PORT"] = str(node_port)
            env["MARBLE_PG_EXPORTER_PORT"] = str(pg_exp_port)
            env["MARBLE_COMPOSE_PROJECT"] = compose_proj

            # Spin up container
            subprocess.run(
                ["docker", "compose", "-p", compose_proj, "-f", "marble/environments/db_env_docker/docker-compose.yml", "up", "-d"],
                env=env,
                capture_output=True,
            )
            import time
            time.sleep(3)

            cmd = [
                uv, "run", "python", "-u", "-m", "marble.experiments.run_benchmark",
                "--benchmark", "database",
                "--manifest", manifest_path,
                "--split", split,
                "--task-ids", str(task_id),
                "--baseline", b_name,
                "--seed", "42",
                "--max-iterations", "5",
                "--retrieval", "visible_k",
                "--max-cards", "5",
                "--lambda", "0.15",
                "--beta", "0.25",
                "--out", rd,
            ]
            try:
                subprocess.run(cmd, env=env, check=True)
            finally:
                subprocess.run(
                    ["docker", "compose", "-p", compose_proj, "-f", "marble/environments/db_env_docker/docker-compose.yml", "down", "-v"],
                    capture_output=True,
                )
        else:
            cmd = [
                uv, "run", "python", "-u", "-m", "marble.experiments.run_benchmark",
                "--benchmark", benchmark,
                "--manifest", manifest_path,
                "--split", split,
                "--task-ids", str(task_id),
                "--baseline", b_name,
                "--seed", "42",
                "--max-iterations", "5",
                "--retrieval", "visible_k",
                "--max-cards", "5",
                "--lambda", "0.15",
                "--beta", "0.25",
                "--out", rd,
            ]
            subprocess.run(cmd, check=True)

        print(f"✅ Successfully backfilled {b_name} | {benchmark} Task {task_id}.")

print("\n🎉 All backfills finished! Re-aggregating Master Table...")
EOF

python3 scripts/aggregate_hard_master_table.py \
    --run-dir runs/ \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out-json "runs/master_table_classical_baselines.json"

echo "======================================================================"
echo "✅ All tables updated in runs/master_table_classical_baselines.json"
echo "======================================================================"
