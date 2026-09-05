#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="${UV:-$(command -v uv || echo "uv")}"
MANIFEST="${MANIFEST:-configs/experiments/multiagentbench_stratified_frozen.json}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-4}"
CKPT="${1:-runs/ours_rl_v2_policy.json}"

OUT_DIR_RL2_FIXED="runs/eval_ours_rl_v2_fixed_test"
mkdir -p "$OUT_DIR_RL2_FIXED"

echo "======================================================================"
echo "⚡ Ours RL Policy Evaluation on Frozen Benchmark"
echo "  Manifest:    $MANIFEST"
echo "  Split:       $SPLIT"
echo "  Seed:        $SEED"
echo "  Checkpoint:  $CKPT"
echo "  Concurrency: $CONCURRENCY Workers"
echo "  Time:        $(date)"
echo "======================================================================"

trap 'docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true' EXIT INT TERM

./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --concurrency "$CONCURRENCY" \
    --controller-checkpoint "$CKPT" \
    --out "$OUT_DIR_RL2_FIXED"

echo "📊 Generating Comparison Summary..."
$UV run python -m marble.experiments.evaluate \
    --run-dir "$OUT_DIR_RL2_FIXED" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" > runs/fixed_rl_v2_table.json
echo "✅ Results saved to runs/fixed_rl_v2_table.json"
