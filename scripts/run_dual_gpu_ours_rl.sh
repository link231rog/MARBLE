#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "========================================================================"
echo "🚀 [Dual-GPU Runner] Launching Ours-RL (test_hard, 24 Tasks) across GPU1 + GPU2"
echo "   Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

export MARBLE_QWEN_API_KEYS="EMPTY,EMPTY"
export MARBLE_QWEN_API_BASES="http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1"
export MARBLE_QWEN_API_MODELS="qwen_rl,qwen_rl"

OUT_DIR="runs/test_hard_ours_rl"
mkdir -p "$OUT_DIR"

echo ">>> Dispatching 24 test_hard tasks across 8 workers load-balanced on GPU1 + GPU2..."
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_rl \
    --split test_hard \
    --concurrency 8 \
    --controller-checkpoint runs/checkpoints/qwen_rl \
    --qwen-api-base "$MARBLE_QWEN_API_BASES" \
    --qwen-api-model qwen_rl \
    --out "$OUT_DIR" \
    2>&1 | tee runs/test_hard_ours_rl_run.log

echo ""
echo "========================================================================"
echo "📊 [Dual-GPU Runner] Generating Master Table 1 Final Report..."
echo "========================================================================"
./.venv/bin/python scripts/aggregate_hard_master_table.py \
    --run-dir runs/hard_* runs/test_hard_ours_sft runs/test_hard_ours_rl \
    > runs/master_table1_final.md 2>&1 || true

cat runs/master_table1_final.md || true

echo ""
echo "========================================================================"
echo "🛑 [Dual-GPU Runner] Freeing VRAM on GPU1 and GPU2 (VRAM Sentinel Protocol)..."
echo "========================================================================"
ssh cis_gpu1.utlab.ltd "bash /data/home/huangzixuan/stop_vllm.sh" 2>/dev/null || true
ssh cis_gpu2.utlab.ltd "bash /data/home/huangzixuan/stop_vllm.sh" 2>/dev/null || true
kill $(lsof -t -i:18000 -i:18001 2>/dev/null) 2>/dev/null || true

echo "========================================================================"
echo "🎉 DUAL-GPU OURS-RL BENCHMARK FULLY COMPLETED!"
echo "   End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
