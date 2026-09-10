#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Code & Config Sync to GPU1
# Mac (Development) -> cis_gpu1.utlab.ltd (Execution & Benchmark Engine)
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REMOTE_HOST="cis_gpu1.utlab.ltd"
REMOTE_DIR="/data/home/huangzixuan/workspace/MARBLE"

echo "======================================================================"
echo "🚀 Syncing MARBLE codebase to $REMOTE_HOST:$REMOTE_DIR..."
echo "======================================================================"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'runs/rl_v3_rollouts' \
    --exclude 'runs/test_hard_*' \
    --exclude 'runs/native_db_w*' \
    --exclude '*.sock' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

echo "======================================================================"
echo "✅ Sync to $REMOTE_HOST completed successfully!"
echo "======================================================================"
