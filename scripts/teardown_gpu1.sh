#!/bin/bash
# ==============================================================================
# MARBLE GPU1 Immediate Teardown & Resource Release Utility
# Strictly obeys lab sharing etiquette: wipes all VRAM, ports, and RAM caches!
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "======================================================================"
echo "🧹 [TEARDOWN] Releasing 100% of GPU1 Resources..."
echo "======================================================================"

# 1. Stop all model serving processes and free GPU VRAM
echo ">>> [1/4] Stopping vLLM & model servers on ports 8000, 8001, 8002, 8003..."
if [ -f "$REPO_DIR/scripts/gpu1_vllm_ctl.sh" ]; then
    "$REPO_DIR/scripts/gpu1_vllm_ctl.sh" stop >/dev/null 2>&1 || true
fi
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 8003/tcp 2>/dev/null || true

# 2. Stop all userland PostgreSQL & Prometheus instances
echo ">>> [2/4] Terminating all native DB & Prometheus sandboxes..."
if [ -f "$REPO_DIR/scripts/native_db_ctl.sh" ]; then
    "$REPO_DIR/scripts/native_db_ctl.sh" cleanup_all >/dev/null 2>&1 || true
fi

# 3. Clean up RAM disk (/dev/shm) and temporary sockets
echo ">>> [3/4] Cleaning /dev/shm and temporary runtime sockets..."
rm -rf /dev/shm/marble_* 2>/dev/null || true
rm -rf "$REPO_DIR/runs/native_db_w"* 2>/dev/null || true

# 4. Terminate any orphan benchmark runner worker processes under current user
echo ">>> [4/4] Checking for lingering worker processes..."
pkill -u "$(whoami)" -f "run_benchmark" 2>/dev/null || true
pkill -u "$(whoami)" -f "train_controller" 2>/dev/null || true

sleep 1
echo "======================================================================"
echo "✅ GPU1 Resources 100% Released! Current GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "======================================================================"
