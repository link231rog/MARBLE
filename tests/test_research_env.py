import json
from datetime import datetime
from types import SimpleNamespace

import arxiv

from marble.environments.research_env import _serialize_paper
from marble.environments.research_utils.paper_collector import Paper


def test_serialize_paper_supports_current_result_and_plain_object():
    result = arxiv.Result(
        entry_id="https://arxiv.org/abs/1234.5678",
        published=datetime(2024, 1, 2),
        title="A paper",
        authors=[arxiv.Result.Author("Alice")],
        summary="An abstract",
    )
    plain_object = SimpleNamespace(title="Plain paper", published=datetime(2024, 1, 2))

    result_data = _serialize_paper(result)
    plain_data = _serialize_paper(plain_object)

    assert result_data["title"] == "A paper"
    assert result_data["authors"] == [{"name": "Alice", "affiliation": []}]
    assert plain_data["title"] == "Plain paper"
    assert json.dumps(result_data)
    assert json.dumps(plain_data)


def test_serialize_paper_keeps_pydantic_paper_schema():
    data = _serialize_paper(Paper(title="A paper", abstract="An abstract"))

    assert data["title"] == "A paper"
    assert data["abstract"] == "An abstract"
    assert "introduction" not in data
