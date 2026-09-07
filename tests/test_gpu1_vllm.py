import unittest
from unittest.mock import patch, MagicMock
from marble.utils.gpu1_vllm import is_vllm_alive, ensure_ssh_tunnel, on_demand_gpu1_vllm


class TestGpu1Vllm(unittest.TestCase):
    def test_is_vllm_alive_false_when_unreachable(self):
        self.assertFalse(is_vllm_alive("http://127.0.0.1:59999/v1", timeout=0.2))

    @patch("subprocess.run")
    def test_ensure_ssh_tunnel_already_listening(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="LISTEN")
        self.assertTrue(ensure_ssh_tunnel("fake_host", 18000))
        # Should not launch another ssh tunnel
        mock_run.assert_called_once()

    @patch("marble.utils.gpu1_vllm.stop_gpu1_vllm")
    @patch("marble.utils.gpu1_vllm.start_gpu1_vllm")
    def test_on_demand_context_manager(self, mock_start, mock_stop):
        mock_start.return_value = "http://127.0.0.1:18000/v1"
        with on_demand_gpu1_vllm() as ep:
            self.assertEqual(ep, "http://127.0.0.1:18000/v1")
        mock_start.assert_called_once()
        mock_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
