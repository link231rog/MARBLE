import json

from marble.controllers.json_controller import VALID_VISIBILITIES
from marble.controllers.qwen_lora import (
    completion_only_collator,
    completion_only_example,
    export_sft_pairs,
    load_qwen_rl_samples,
    make_qwen_lora_controller,
    train_qwen_rl,
    train_qwen_sft,
    weighted_completion_loss,
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


def test_export_sft_pairs_uses_recorded_prompt_snapshot(tmp_path):
    trace = tmp_path / "t.jsonl"
    prompt = "[TASK]\nreal goal\n[ACTIVE MEMORY INDEX]\nm0: prior finding"
    ev = {
        "event": "memory_decision", "memory_id": "m1",
        "controller_prompt": prompt,
        "proposal": {"proposal_id": "p1", "task_id": "t", "agent_id": "a1",
                     "source": "worker", "title": "shared plan", "raw_value": "v",
                     "step_index": 1},
        "target": {"visibility": "private", "supersedes": "m0"},
    }
    trace.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    pairs = export_sft_pairs([str(trace)])

    assert pairs == [
        (prompt, '{"visibility": "private", "supersedes": "m0"}')
    ]


def test_export_sft_pairs_keeps_absent_decisions(tmp_path):
    trace = tmp_path / "t.jsonl"
    ev = {
        "event": "memory_decision",
        "controller_prompt": "recorded prompt",
        "proposal": {"proposal_id": "p1", "task_id": "t", "agent_id": "a1",
                     "source": "worker", "title": "noise", "raw_value": "ignore",
                     "step_index": 1},
        "target": {"visibility": "absent", "supersedes": None},
    }
    trace.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    assert export_sft_pairs([str(trace)]) == [
        ("recorded prompt", '{"visibility": "absent", "supersedes": null}')
    ]


def test_train_qwen_sft_requires_torch():
    # torch/transformers are not installed offline; the function must fail loudly
    import pytest

    with pytest.raises(Exception):
        train_qwen_sft([("prompt", '{"visibility":"global","supersedes":null}')],
                       "/tmp/out", "Qwen/Qwen3-4B-Instruct-2507")


def test_completion_only_example_masks_prompt_tokens():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1, len(text) + 1))}

    example = completion_only_example(Tokenizer(), "prompt", "{}", max_len=16)
    prompt_len = len("prompt\n")
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert example["labels"][prompt_len:] == [1, 2, 99]
    assert example["attention_mask"] == [1] * len(example["input_ids"])


def test_completion_only_example_drops_prompt_when_completion_fills_budget():
    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1, len(text) + 1))}

    example = completion_only_example(Tokenizer(), "long prompt", "{}", max_len=2)
    assert example["input_ids"] == [1, 2]
    assert example["labels"] == [1, 2]


def test_train_qwen_sft_rejects_empty_pairs_before_model_import():
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        train_qwen_sft([], "/tmp/out", "not-loaded")


def test_load_qwen_rl_samples_uses_same_task_memory_credit(tmp_path):
    trace = tmp_path / "t1.jsonl"
    second_trace = tmp_path / "t2.jsonl"
    decisions = [
        {
            "event": "memory_decision",
            "memory_id": "m1",
            "controller_prompt": "first prompt",
            "controller_output": '{"visibility":"global","supersedes":null}',
            "proposal": {"task_id": "same", "agent_id": "a", "raw_value": "fact"},
            "target": {"visibility": "global", "supersedes": None},
        },
        {
            "event": "memory_read",
            "memory_id": "m1",
            "reader_id": "b",
        },
        {
            "event": "memory_decision",
            "controller_prompt": "second prompt",
            "proposal": {"task_id": "same", "agent_id": "a", "raw_value": "ignored"},
            "target": {"visibility": "global", "supersedes": None},
        },
    ]
    trace.write_text(
        "".join(json.dumps(decision) + "\n" for decision in decisions),
        encoding="utf-8",
    )
    second_trace.write_text(
        json.dumps(
            {
                "event": "memory_decision",
                "memory_id": "m2",
                "controller_prompt": "second rollout",
                "controller_output": '{"visibility":"global","supersedes":null}',
                "proposal": {"task_id": "same", "agent_id": "a", "raw_value": "fact"},
                "target": {"visibility": "global", "supersedes": None},
            }
        ) + "\n",
        encoding="utf-8",
    )

    samples, stats = load_qwen_rl_samples(
        [str(trace), str(second_trace)], [2.0, 0.0]
    )

    assert samples == [{
        "prompt": "first prompt",
        "completion": '{"visibility":"global","supersedes":null}',
        "advantage": 1.25 - 0.05 / 4096,
    }, {
        "prompt": "second rollout",
        "completion": '{"visibility":"global","supersedes":null}',
        "advantage": -0.05 / 4096,
    }]
    assert stats["decisions"] == 3
    assert stats["skipped_no_credit"] == 1


def test_train_qwen_rl_rejects_empty_replay_before_model_import(tmp_path):
    import pytest

    trace = tmp_path / "empty.jsonl"
    trace.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no replayable"):
        train_qwen_rl([str(trace), str(trace)], "/tmp/out", "not-loaded", rewards=[1.0, 1.0])


def test_completion_only_collator_and_weighted_loss():
    import pytest

    torch = pytest.importorskip("torch")
    batch = completion_only_collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 2],
                "sample_weight": 1.0,
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 1],
                "sample_weight": 0.0,
            },
        ],
        pad_token_id=0,
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 2], [-100, 1, -100]]
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    weighted = weighted_completion_loss(logits, batch["labels"], batch["sample_weight"])
    first_only = weighted_completion_loss(
        logits[:1], batch["labels"][:1], torch.tensor([1.0])
    )
    assert torch.isclose(weighted, first_only)
