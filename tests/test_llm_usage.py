import unittest
from types import SimpleNamespace

from marble.llms import ApiUsageMeter
from marble.llms.usage import record_successful_completion


class TestApiUsageMeter(unittest.TestCase):
    def test_nested_meters_record_successful_completion(self) -> None:
        completion = SimpleNamespace(
            usage={"input_tokens": 11, "output_tokens": 4},
        )

        with ApiUsageMeter() as outer:
            with ApiUsageMeter() as inner:
                record_successful_completion(completion)

        expected = {"api_calls": 1, "input_tokens": 11, "output_tokens": 4}
        self.assertEqual(outer.snapshot(), expected)
        self.assertEqual(inner.snapshot(), expected)

    def test_missing_usage_is_recorded_as_zero_tokens(self) -> None:
        with ApiUsageMeter() as meter:
            record_successful_completion(SimpleNamespace(usage=None))

        self.assertEqual(
            meter.snapshot(),
            {"api_calls": 1, "input_tokens": 0, "output_tokens": 0},
        )


if __name__ == "__main__":
    unittest.main()
