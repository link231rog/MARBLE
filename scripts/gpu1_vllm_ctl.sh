#!/bin/bash
# GPU1 vLLM Management Utility (Minimal VRAM on-demand service)

ACTION="${1:-status}"
REMOTE="cis_gpu1.utlab.ltd"
TUNNEL_PORT=18000

case "$ACTION" in
  start)
    echo "Starting minimal VRAM vLLM on $REMOTE..."
    ssh -o ConnectTimeout=10 "$REMOTE" 'bash /data/home/huangzixuan/start_vllm_minimal.sh'
    # Check tunnel
    if ! lsof -i:$TUNNEL_PORT | grep LISTEN > /dev/null; then
      echo "Opening local SSH tunnel to port $TUNNEL_PORT..."
      ssh -fNT -L 127.0.0.1:$TUNNEL_PORT:localhost:8000 "$REMOTE"
    fi
    echo "Waiting for endpoint http://127.0.0.1:$TUNNEL_PORT/v1/models..."
    for i in $(seq 1 30); do
      if curl -s "http://127.0.0.1:$TUNNEL_PORT/v1/models" > /dev/null; then
        echo "vLLM is READY on http://127.0.0.1:$TUNNEL_PORT/v1"
        exit 0
      fi
      sleep 2
    done
    echo "Timeout waiting for vLLM to initialize."
    exit 1
    ;;
  stop)
    echo "Stopping vLLM on $REMOTE to free 100% GPU memory..."
    ssh -o ConnectTimeout=10 "$REMOTE" 'bash /data/home/huangzixuan/stop_vllm.sh'
    echo "Done."
    ;;
  status)
    echo "=== GPU Status on $REMOTE ==="
    ssh -o ConnectTimeout=10 "$REMOTE" 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader'
    echo "=== Local Tunnel (port $TUNNEL_PORT) ==="
    lsof -i:$TUNNEL_PORT | grep LISTEN || echo "Tunnel not running"
    echo "=== vLLM Endpoint Health ==="
    curl -s --connect-timeout 2 "http://127.0.0.1:$TUNNEL_PORT/v1/models" || echo "vLLM endpoint inactive"
    echo ""
    ;;
  download-status)
    echo "=== Qwen3.5-4B Download Status on $REMOTE ==="
    ssh -o ConnectTimeout=10 "$REMOTE" '
      TARGET="/data/home/huangzixuan/models/Qwen3.5-4B"
      if [ -f "$TARGET/model.safetensors-00001-of-00002.safetensors" ] && [ -f "$TARGET/model.safetensors-00002-of-00002.safetensors" ]; then
        echo "Status: COMPLETE (All shards downloaded and ready)"
      else
        echo "Status: DOWNLOADING..."
      fi
      ls -lh "$TARGET"/*.incomplete 2>/dev/null || true
      du -sh "$TARGET" 2>/dev/null || true
      echo "--- Recent Log ---"
      tail -n 10 /data/home/huangzixuan/download_qwen3.5.log 2>/dev/null || true
    '
    ;;
  *)
    echo "Usage: $0 {start|stop|status|download-status}"
    exit 1
    ;;
esac
