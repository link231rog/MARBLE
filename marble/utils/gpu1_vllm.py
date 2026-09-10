"""On-demand vLLM manager for cis_gpu1.utlab.ltd.

Provides lifecycle controls (start, stop, health-check) and a context manager
to ensure vLLM runs on gpu1 only when evaluating 'ours' SFT / RL policies,
using minimal VRAM (~18GB out of 96GB on H20), and stopping immediately
afterwards so GPU memory is 100% freed for others.
"""

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_HOST = os.environ.get("MARBLE_GPU1_HOST", "cis_gpu1.utlab.ltd")
DEFAULT_TUNNEL_PORT = int(os.environ.get("MARBLE_GPU1_TUNNEL_PORT", "18000"))
DEFAULT_LOCAL_ENDPOINT = f"http://127.0.0.1:{DEFAULT_TUNNEL_PORT}/v1"


def is_vllm_alive(endpoint: str = DEFAULT_LOCAL_ENDPOINT, timeout: float = 2.0) -> bool:
    """Check if the vLLM OpenAI-compatible endpoint responds."""
    models_url = f"{endpoint.rstrip('/')}/models"
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "MARBLE-HealthCheck"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return bool(response.status == 200)
    except Exception:
        return False


def ensure_ssh_tunnel(remote_host: str = DEFAULT_REMOTE_HOST, tunnel_port: int = DEFAULT_TUNNEL_PORT) -> bool:
    """Ensure SSH tunnel mapping localhost:tunnel_port to remote localhost:8000 is active."""
    # Check if port is already open
    check = subprocess.run(
        ["lsof", f"-i:{tunnel_port}"],
        capture_output=True,
        text=True
    )
    if check.returncode == 0 and "LISTEN" in check.stdout:
        return True

    logger.info(f"Opening SSH tunnel to {remote_host}: -L 127.0.0.1:{tunnel_port}:localhost:8000")
    try:
        subprocess.run(
            ["ssh", "-fNT", "-L", f"127.0.0.1:{tunnel_port}:localhost:8000", remote_host],
            check=True,
            timeout=10,
        )
        time.sleep(1)
        return True
    except Exception as e:
        logger.error(f"Failed to open SSH tunnel to {remote_host}: {e}")
        return False


def start_gpu1_vllm(
    remote_host: str = DEFAULT_REMOTE_HOST,
    tunnel_port: int = DEFAULT_TUNNEL_PORT,
    timeout: int = 75,
) -> str:
    """Start minimal VRAM vLLM on gpu1 and wait until /v1/models is ready."""
    ensure_ssh_tunnel(remote_host, tunnel_port)
    endpoint = f"http://127.0.0.1:{tunnel_port}/v1"

    if is_vllm_alive(endpoint):
        logger.info(f"vLLM on {remote_host} is already running and healthy at {endpoint}.")
        return endpoint

    logger.info(f"Starting minimal VRAM vLLM on {remote_host}...")
    start_cmd = ["ssh", "-o", "ConnectTimeout=10", remote_host, "bash /data/home/huangzixuan/start_vllm_minimal.sh"]
    res = subprocess.run(start_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to trigger start_vllm_minimal.sh on {remote_host}: {res.stderr}")

    logger.info("Waiting for vLLM API server to initialize (loading weights & KV cache)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_vllm_alive(endpoint):
            elapsed = time.time() - start_time
            logger.info(f"vLLM on {remote_host} is ready at {endpoint} (took {elapsed:.1f}s).")
            return endpoint
        time.sleep(2.5)

    raise TimeoutError(
        f"vLLM on {remote_host} failed to become healthy within {timeout}s at {endpoint}."
    )


def stop_gpu1_vllm(remote_host: str = DEFAULT_REMOTE_HOST) -> None:
    """Stop vLLM on gpu1 immediately to release 100% of GPU memory."""
    logger.info(f"Stopping vLLM on {remote_host} to free GPU memory...")
    stop_cmd = ["ssh", "-o", "ConnectTimeout=10", remote_host, "bash /data/home/huangzixuan/stop_vllm.sh"]
    try:
        subprocess.run(stop_cmd, capture_output=True, text=True, timeout=15)
        logger.info(f"vLLM on {remote_host} stopped successfully.")
    except Exception as e:
        logger.warning(f"Error stopping vLLM on {remote_host}: {e}")


@contextmanager
def on_demand_gpu1_vllm(
    remote_host: str = DEFAULT_REMOTE_HOST,
    tunnel_port: int = DEFAULT_TUNNEL_PORT,
    timeout: int = 75,
) -> Iterator[str]:
    """Context manager that starts vLLM on gpu1 on enter and ensures it stops on exit."""
    endpoint = start_gpu1_vllm(remote_host=remote_host, tunnel_port=tunnel_port, timeout=timeout)
    try:
        yield endpoint
    finally:
        stop_gpu1_vllm(remote_host=remote_host)
