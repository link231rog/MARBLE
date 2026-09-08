import unittest
from unittest.mock import MagicMock, patch

from marble.llms.text_embedding import text_embedding


class TestTextEmbedding(unittest.TestCase):
    @patch("marble.llms.text_embedding.litellm.embedding")
    def test_text_embedding(self, mock_litellm_embedding) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_litellm_embedding.return_value = mock_resp

        content = "This is a test sentence."
        embedding = text_embedding(
            model="text-embedding-3-small",
            input=content,
        )
        self.assertIsInstance(embedding, list)
        for entry in embedding:
            self.assertIsInstance(entry, float)


if __name__ == "__main__":
    unittest.main()
