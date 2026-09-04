#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Communication Governor (Context Pruning) Ablation Pipeline
# Runs ours_sft with CommunicationGovernor (comm_gov=on) and ours_rl_v2 (fixed)
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
SPLIT="test"

cleanup_docker() {
    echo "Cleaning up lingering Docker containers..."
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

echo "======================================================================"
echo "🧪 Starting Context Pruning (CommunicationGovernor) Ablation Suite"
echo "  Split:       $SPLIT"
echo "  Concurrency: 16 Workers"
echo "======================================================================"

cleanup_docker

# 1. Ablation: ours_sft with Context Pruning Active (comm_gov:on)
echo ""
echo "======================================================================"
echo "[Ablation 1/2] ⚡ Ours SFT + CommunicationGovernor (comm_gov=on)"
echo "Time: $(date)"
echo "======================================================================"
OUT_DIR_COMM="runs/eval_ablation_comm_gov_on_sft_test"
mkdir -p "$OUT_DIR_COMM"

./scripts/run_parallel_benchmark.sh \
    --baseline ours_sft \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint runs/ours_sft_policy.json \
    --enable-comm-governor \
    --out "$OUT_DIR_COMM"

cleanup_docker
sleep 5

# 2. Re-run: ours_rl_v2 with Template Placeholder Fix Applied
echo ""
echo "======================================================================"
echo "[Validation 2/2] ⚡ Ours RL v2 with Placeholder Leak Fix Applied"
echo "Time: $(date)"
echo "======================================================================"
OUT_DIR_RL2_FIXED="runs/eval_ours_rl_v2_fixed_test"
mkdir -p "$OUT_DIR_RL2_FIXED"

./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint runs/ours_rl_v2_policy.json \
    --out "$OUT_DIR_RL2_FIXED"

cleanup_docker
sleep 5

echo ""
echo "======================================================================"
echo "🎉 CONTEXT PRUNING ABLATION & FIXED RL V2 EVALUATION COMPLETED!"
echo "======================================================================"

echo "📊 Generating Comparison Summary..."
$UV run python -m marble.experiments.evaluate --run-dir "$OUT_DIR_COMM" | tee runs/ablation_comm_gov_table.json
$UV run python -m marble.experiments.evaluate --run-dir "$OUT_DIR_RL2_FIXED" | tee runs/fixed_rl_v2_table.json

echo "✅ Results successfully saved to runs/ablation_comm_gov_table.json and runs/fixed_rl_v2_table.json"
