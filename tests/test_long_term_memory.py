import unittest
from unittest.mock import patch

from litellm.types.utils import Message
from numpy import ndarray

from marble.memory.long_term_memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.patcher_embed = patch("marble.memory.long_term_memory.text_embedding")
        self.mock_embed = self.patcher_embed.start()
        def _embed(model, input):
            text = str(input).lower()
            if "cold" in text or "weather" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]
        self.mock_embed.side_effect = _embed

        self.patcher_model = patch("marble.memory.long_term_memory.model_prompting")
        self.mock_model = self.patcher_model.start()
        self.mock_model.return_value = [Message(content="weather summary", role="assistant")]

        self.memory = LongTermMemory()

    def tearDown(self) -> None:
        self.patcher_embed.stop()
        self.patcher_model.stop()

    def test_update(self) -> None:
        result_1 = Message(content="It is cold today.", role="assistant")
        self.memory.update(
            key="key", information={"type": "action_response", "result": result_1}
        )
        result_2 = Message(content="I want to sleep.", role="assistant")
        self.memory.update(
            key="key", information={"type": "action_response", "result": result_2}
        )
        self.assertIsInstance(self.memory.storage, list)
        self.assertIsInstance(self.memory.storage[0], tuple)
        self.assertIsInstance(self.memory.storage[0][0], dict)
        self.assertIsInstance(self.memory.storage[0][1], ndarray)

    def test_retrieve_latest(self) -> None:
        result_1 = Message(content="It is cold today.", role="assistant")
        information_1 = {"type": "action_response", "result": result_1}
        self.memory.update(key="key", information=information_1)
        result_2 = Message(content="I want to sleep.", role="assistant")
        information_2 = {"type": "action_response", "result": result_2}
        self.memory.update(key="key", information=information_2)
        true_latest_information = information_2
        latest_information = self.memory.retrieve_latest()
        self.assertIs(latest_information, true_latest_information)

    def test_retrieve_most_relevant(self) -> None:
        result_1 = Message(content="It is cold today.", role="assistant")
        information_1 = {"type": "action_response", "result": result_1}
        self.memory.update(key="key", information=information_1)
        result_2 = Message(content="I want to sleep.", role="assistant")
        information_2 = {"type": "action_response", "result": result_2}
        self.memory.update(key="key", information=information_2)
        true_relevant_information = information_1
        result = Message(content="What's the weather like today?", role="assistant")
        most_relevant_information = self.memory.retrieve_most_relevant(
            information={"type": "action_response", "result": result}
        )
        self.assertIs(most_relevant_information[0], true_relevant_information)

        most_relevant_information = self.memory.retrieve_most_relevant(
            summarize=True, information={"type": "action_response", "result": result}
        )
        self.assertIsInstance(most_relevant_information, Message)

    def test_retrieve_all(self) -> None:
        result_1 = Message(content="It is cold today.", role="assistant")
        information_1 = {"type": "action_response", "result": result_1}
        self.memory.update(key="key", information=information_1)
        result_2 = Message(content="I want to sleep.", role="assistant")
        information_2 = {"type": "action_response", "result": result_2}
        self.memory.update(key="key", information=information_2)
        all_information = self.memory.retrieve_all()
        self.assertIn(information_1, all_information)
        self.assertIn(information_2, all_information)


if __name__ == "__main__":
    unittest.main()
