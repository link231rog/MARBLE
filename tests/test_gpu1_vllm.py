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

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_ensure_ssh_tunnel_binds_to_loopback(self, mock_run, mock_sleep):
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),
            MagicMock(returncode=0, stdout=""),
        ]
        self.assertTrue(ensure_ssh_tunnel("fake_host", 18000))
        self.assertEqual(mock_run.call_count, 2)
        ssh_call_args = mock_run.call_args_list[1][0][0]
        self.assertIn("127.0.0.1:18000:localhost:8000", ssh_call_args)
        self.assertNotIn("0.0.0.0:18000:localhost:8000", ssh_call_args)

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
