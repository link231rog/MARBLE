import unittest
from types import SimpleNamespace
from unittest.mock import patch

from litellm.types.utils import Message

from marble.llms import ApiUsageMeter
from marble.llms.model_prompting import model_prompting


class TestModelPrompting(unittest.TestCase):
    def test_model_prompting_records_completion_usage(self) -> None:
        prompt = "This is a test sentence."
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=Message(content="Done.", role="assistant"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )
        with patch("marble.llms.model_prompting.litellm.completion", return_value=completion):
            with ApiUsageMeter() as meter:
                message = model_prompting(
                    llm_model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": prompt}],
                )[0]

        self.assertIsInstance(message, Message)
        self.assertEqual(
            meter.snapshot(),
            {"api_calls": 1, "input_tokens": 7, "output_tokens": 3},
        )

    def test_model_prompting_does_not_record_failed_call(self) -> None:
        with patch(
            "marble.llms.model_prompting.litellm.completion",
            side_effect=RuntimeError("provider failed"),
        ):
            with ApiUsageMeter() as meter:
                with self.assertRaises(RuntimeError):
                    model_prompting(
                        llm_model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "hello"}],
                        mode="TEST",
                    )

        self.assertEqual(
            meter.snapshot(),
            {"api_calls": 0, "input_tokens": 0, "output_tokens": 0},
        )

    def test_model_prompting_records_nested_meters_and_missing_usage(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=Message(content="Done.", role="assistant"))],
            usage=None,
        )
        with patch("marble.llms.model_prompting.litellm.completion", return_value=completion):
            with ApiUsageMeter() as outer:
                with ApiUsageMeter() as inner:
                    model_prompting(
                        llm_model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "hello"}],
                    )

        expected = {"api_calls": 1, "input_tokens": 0, "output_tokens": 0}
        self.assertEqual(outer.snapshot(), expected)
        self.assertEqual(inner.snapshot(), expected)

    def test_model_prompting_passes_reasoning_effort(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=Message(content="OK", role="assistant"))],
            usage=None,
        )
        with patch(
            "marble.llms.model_prompting.litellm.completion",
            return_value=completion,
        ) as completion_mock:
            model_prompting(
                llm_model="openai/deepseek-v4-flash",
                messages=[{"role": "user", "content": "hello"}],
                reasoning_effort="none",
            )

        self.assertEqual(completion_mock.call_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(
            completion_mock.call_args.kwargs["allowed_openai_params"],
            ["reasoning_effort"],
        )

    def test_zai_normalizes_messages_without_dropping_tool_calls(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=Message(content="OK", role="assistant"))],
            usage=None,
        )
        assistant = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    type="function",
                    function=SimpleNamespace(name="lookup", arguments='{"q":"x"}'),
                    extra="drop",
                )
            ],
            extra="drop",
        )
        tool = SimpleNamespace(
            role="tool",
            content=None,
            tool_call_id="call-1",
            name="lookup",
            extra="drop",
        )
        with patch(
            "marble.llms.model_prompting.litellm.completion",
            return_value=completion,
        ) as completion_mock, patch.dict(
            "os.environ", {"MARBLE_REASONING_EFFORT": "high"}, clear=False
        ):
            model_prompting(
                llm_model="openai/glm-4.7-flash",
                messages=[assistant, tool],
            )

        kwargs = completion_mock.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://api.z.ai/api/paas/v4")
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "assistant", "content": "", "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }
                ]},
                {"role": "tool", "content": "", "tool_call_id": "call-1"},
            ],
        )
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("allowed_openai_params", kwargs)


if __name__ == "__main__":
    unittest.main()
