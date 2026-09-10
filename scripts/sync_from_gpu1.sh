#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Results & Checkpoints Pull from GPU1
# cis_gpu1.utlab.ltd -> Mac (Analysis & Version Control)
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REMOTE_HOST="cis_gpu1.utlab.ltd"
REMOTE_DIR="/data/home/huangzixuan/workspace/MARBLE"

echo "======================================================================"
echo "📥 Pulling MARBLE results & checkpoints from $REMOTE_HOST..."
echo "======================================================================"

mkdir -p runs/checkpoints

# Sync checkpoints
echo ">>> Syncing trained checkpoints..."
rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/runs/checkpoints/" "runs/checkpoints/" || true

# Sync evaluation runs and tables
echo ">>> Syncing evaluation results and JSON tables..."
rsync -avz --progress \
    --include='table*.json' \
    --include='*.json' \
    --include='*.csv' \
    --include='*/' \
    --include='*summary.json' \
    --include='*metrics.json' \
    --exclude='*.log' \
    --exclude='*memory_trace.jsonl' \
    "$REMOTE_HOST:$REMOTE_DIR/runs/" "runs/" || true

echo "======================================================================"
echo "✅ Pull from $REMOTE_HOST completed!"
echo "======================================================================"
