import os
import unittest
from unittest.mock import patch

from marble.llms.error_handler import api_calling_error_exponential_backoff


class RateLimitError(Exception):
    status_code = 429


class TestApiCallingBackoff(unittest.TestCase):
    def test_429_uses_configured_base_wait_and_keeps_retrying(self) -> None:
        calls = 0

        @api_calling_error_exponential_backoff(retries=2, base_wait_time=1)
        def completion() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("rpm exhausted")
            return "ok"

        with patch.dict(os.environ, {"MARBLE_API_429_BASE_WAIT_TIME": "60"}):
            with patch("marble.llms.error_handler.time.sleep") as sleep:
                self.assertEqual(completion(), "ok")

        self.assertEqual(calls, 2)
        sleep_arg = sleep.call_args[0][0]
        self.assertGreaterEqual(sleep_arg, 60.0)


if __name__ == "__main__":
    unittest.main()
