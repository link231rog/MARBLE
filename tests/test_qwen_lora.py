import json

from marble.controllers.json_controller import VALID_VISIBILITIES
from marble.controllers.qwen_lora import (
    export_sft_pairs,
    make_qwen_lora_controller,
    train_qwen_sft,
)
from marble.memory.schema import MemoryProposal


def test_api_controller_decides_valid():
    ctrl = make_qwen_lora_controller(
        lambda p: '{"visibility": "global", "supersedes": null}'
    )
    out = ctrl.decide(
        MemoryProposal(proposal_id="p1", task_id="t", agent_id="a1",
                      source="worker", title="shared result", raw_value="x",
                      step_index=1),
        [],
    )
    assert out.visibility == "global" and out.exists


def test_export_sft_pairs_one_per_decision(tmp_path):
    trace = tmp_path / "t.jsonl"
    ev = {
        "event": "memory_decision", "memory_id": "m1",
        "proposal": {"proposal_id": "p1", "task_id": "t", "agent_id": "a1",
                     "source": "worker", "title": "shared plan", "raw_value": "v",
                     "step_index": 1},
        "target": {"visibility": "global", "supersedes": None},
    }
    trace.write_text(json.dumps(ev) + "\n", encoding="utf-8")
    pairs = export_sft_pairs([str(trace)])
    assert len(pairs) == 1
    assert json.loads(pairs[0][1]) == {"visibility": "global", "supersedes": None}
    assert "[PROPOSAL]" in pairs[0][0]


def test_train_qwen_sft_requires_torch():
    # torch/transformers are not installed offline; the function must fail loudly
    import pytest

    with pytest.raises(Exception):
        train_qwen_sft([("prompt", '{"visibility":"global","supersedes":null}')],
                       "/tmp/out", "Qwen/Qwen3-4B-Instruct-2507")
