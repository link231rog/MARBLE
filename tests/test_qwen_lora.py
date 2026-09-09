import json
import sys
from types import SimpleNamespace

from marble.controllers.json_controller import VALID_VISIBILITIES
from marble.controllers.qwen_lora import (
    audit_sft_distribution,
    completion_only_collator,
    completion_only_example,
    export_sft_pairs,
    load_qwen_rl_samples,
    make_qwen_lora_controller,
    train_qwen_rl,
    train_qwen_sft,
    weighted_completion_loss,
)
from marble.experiments.ablations import controller_kwargs
from marble.llms import ApiUsageMeter
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


def test_api_generate_fn_records_completion_usage(monkeypatch):
    from marble.controllers.qwen_lora import api_generate_fn

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )

    class Completions:
        def create(self, **kwargs):
            return completion

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    generate = api_generate_fn("https://example.test/v1", "key", "qwen")

    with ApiUsageMeter() as meter:
        assert generate("prompt") == "answer"

    assert meter.snapshot() == {
        "api_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
    }


def test_api_generate_fn_passes_timeout(monkeypatch):
    from marble.controllers.qwen_lora import api_generate_fn

    received = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            received.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    api_generate_fn("https://example.test/v1", "key", "qwen", timeout=45.0)
    assert received.get("timeout") == 45.0


def test_factory_applies_schema_drop_kwargs_to_runtime_prompt():
    prompts = []
    ctrl = make_qwen_lora_controller(
        lambda prompt: prompts.append(prompt) or '{"visibility": ["a1"], "supersedes": null}',
        agent_role_map={"a1": "developer"},
        **controller_kwargs("schema_field", "title"),
    )

    out = ctrl.decide(
        MemoryProposal(
            proposal_id="p1",
            task_id="t",
            agent_id="a1",
            source="worker",
            title="hidden title",
            raw_value="kept value",
            step_index=1,
        ),
        [],
    )

    assert "title: hidden title" not in prompts[0]
    assert json.loads(ctrl.last_raw) == {
        "visibility": ["a1"],
        "supersedes": None,
    }
    assert out.visibility == "targeted" and out.target_recipients == ("a1",) and out.supersedes is None


def test_factory_applies_input_drop_kwargs_to_runtime_prompt():
    prompts = []
    ctrl = make_qwen_lora_controller(
        lambda prompt: prompts.append(prompt) or '{"visibility": "global", "supersedes": null}',
        **controller_kwargs("input", "no_agent_tag"),
    )

    ctrl.decide(
        MemoryProposal(
            proposal_id="p1",
            task_id="t",
            agent_id="a1",
            source="worker",
            title="shared result",
            raw_value="x",
            step_index=1,
        ),
        [],
    )

    assert "agent_reference: a1" not in prompts[0]


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


def test_train_qwen_sft_requires_torch(monkeypatch):
    import builtins
    import pytest

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "transformers", "peft"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        train_qwen_sft(
            [("prompt", '{"visibility":"global","supersedes":null}')],
            "/tmp/out",
            "dummy-model",
        )


def test_train_qwen_sft_disallows_unauthorized_remote_download(monkeypatch):
    import pytest

    monkeypatch.delenv("MARBLE_ALLOW_HF_DOWNLOAD", raising=False)
    with pytest.raises(Exception):
        train_qwen_sft(
            [("prompt", '{"visibility":"global","supersedes":null}')],
            "/tmp/out",
            "nonexistent-remote-repo/model",
        )


def test_train_qwen_sft_pipeline(monkeypatch, tmp_path):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10, 11, 12]}

        def save_pretrained(self, path):
            pass

    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=None)

        def save_pretrained(self, path):
            pass

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: FakeTokenizer()
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: FakeModel(),
    )
    monkeypatch.setattr("peft.get_peft_model", lambda model, lora: model)
    monkeypatch.setattr("transformers.Trainer.train", lambda self: None)

    out_dir = str(tmp_path / "adapter")
    res = train_qwen_sft(
        [("prompt", '{"visibility":"global","supersedes":null}')],
        out_dir,
        "dummy/model",
        epochs=1,
    )
    assert res == out_dir
    assert (tmp_path / "adapter" / "adapter_metadata.json").is_file()


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
            "proposal": {"task_id": "same", "agent_id": "b", "raw_value": "ignored"},
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
        "task_advantage": 1.0,
        "benchmark": "",
        "task_id": "same",
    }, {
        "prompt": "second rollout",
        "completion": '{"visibility":"global","supersedes":null}',
        "advantage": -0.05 / 4096,
        "task_advantage": -1.0,
        "benchmark": "",
        "task_id": "same",
    }]
    assert stats["decisions"] == 3
    assert stats["skipped_no_credit"] == 1


def test_load_qwen_rl_samples_keeps_same_task_id_per_benchmark(tmp_path):
    traces = []
    rewards = []
    for benchmark, benchmark_scores in (("coding", (2.0, 0.0)), ("research", (10.0, 0.0))):
        for index, reward in enumerate(benchmark_scores):
            trace_dir = tmp_path / benchmark / str(index)
            trace_dir.mkdir(parents=True)
            trace = trace_dir / "memory_trace.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        json.dumps({
                            "event": "memory_decision",
                            "memory_id": f"{benchmark}-{index}",
                            "controller_prompt": f"{benchmark}-{index}",
                            "controller_output": "{}",
                            "proposal": {"task_id": "same", "agent_id": "a", "raw_value": "fact"},
                            "target": {"visibility": "global", "supersedes": None},
                        }),
                        json.dumps({
                            "event": "memory_read",
                            "memory_id": f"{benchmark}-{index}",
                            "reader_id": "b",
                        }),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            (trace_dir / "summary.json").write_text(
                json.dumps({"benchmark": benchmark, "score_status": "available"}),
                encoding="utf-8",
            )
            traces.append(str(trace))
            rewards.append(reward)

    samples, _ = load_qwen_rl_samples(traces, rewards)

    assert samples[0]["advantage"] == 1.25 - 0.05 / 4096
    # Chapter 4 §5.1: std normalization achieves scale invariance across tasks
    assert samples[2]["advantage"] == 1.25 - 0.05 / 4096


def test_load_qwen_rl_samples_excludes_unavailable_score_trace(tmp_path):
    available_dir = tmp_path / "available"
    unavailable_dir = tmp_path / "unavailable"
    available_dir.mkdir()
    unavailable_dir.mkdir()
    available = available_dir / "memory_trace.jsonl"
    unavailable = unavailable_dir / "memory_trace.jsonl"
    event = {
        "event": "memory_decision",
        "memory_id": "m1",
        "controller_prompt": "prompt",
        "controller_output": "{}",
        "proposal": {"task_id": "same", "agent_id": "a", "raw_value": "fact"},
        "target": {"visibility": "global", "supersedes": None},
    }
    for trace in (available, unavailable):
        trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
    (available_dir / "summary.json").write_text(
        json.dumps({"benchmark": "coding", "score_status": "available"}),
        encoding="utf-8",
    )
    (unavailable_dir / "summary.json").write_text(
        json.dumps({"benchmark": "coding", "score_status": "unavailable"}),
        encoding="utf-8",
    )

    samples, stats = load_qwen_rl_samples(
        [str(available), str(unavailable), str(available)], [1.0, 100.0, 1.0]
    )

    assert len(samples) == 2
    assert stats["decisions"] == 2


def test_load_qwen_rl_samples_undersized_tasks_raises_value_error(tmp_path):
    import pytest

    trace_dir = tmp_path / "coding" / "0"
    trace_dir.mkdir(parents=True)
    trace = trace_dir / "memory_trace.jsonl"
    trace.write_text(
        json.dumps({
            "event": "memory_decision",
            "memory_id": "m1",
            "proposal": {"task_id": "task42", "agent_id": "a", "raw_value": "fact"},
            "target": {"visibility": "global", "supersedes": None},
        }) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "summary.json").write_text(
        json.dumps({"benchmark": "coding", "score_status": "available"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"undersized tasks: coding:task42"):
        load_qwen_rl_samples([str(trace)], [1.0])


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


def test_audit_sft_distribution():
    from marble.controllers.qwen_lora import audit_sft_distribution

    pairs = [
        ("prompt1", '{"visibility": "global", "supersedes": null}'),
        ("prompt2", '{"visibility": "private", "supersedes": null}'),
        ("prompt3", '{"visibility": "absent", "supersedes": null}'),
        ("prompt4", '{"visibility": "global", "supersedes": "m1"}'),
        ("prompt5", 'invalid json'),
    ]
    report = audit_sft_distribution(pairs)
    assert report["total_pairs"] == 5
    assert report["counts"]["global"] == 2
    assert report["counts"]["private"] == 1
    assert report["counts"]["absent"] == 1
    assert report["counts"]["invalid"] == 1
    assert report["proportions"]["global"] == 0.4


def test_train_qwen_rl_trajectory_grpo_pipeline(monkeypatch, tmp_path):
    pytest = pytest_module = __import__("pytest")
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from types import SimpleNamespace

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10, 11, 12]}

        def save_pretrained(self, path):
            pass

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=None)
            self.dummy_param = nn.Parameter(torch.zeros(2, requires_grad=True))

        def save_pretrained(self, path):
            pass

        def forward(self, *a, **kw):
            in_ids = kw.get("input_ids")
            if in_ids is None and a:
                in_ids = a[0]
            seq_len = in_ids.shape[1] if in_ids is not None else 7
            dev = in_ids.device if in_ids is not None else "cpu"
            return SimpleNamespace(
                logits=torch.zeros((1, seq_len, 20), requires_grad=True, device=dev)
            )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: FakeTokenizer()
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: FakeModel(),
    )
    monkeypatch.setattr("peft.get_peft_model", lambda model, lora: model)

    trace1 = tmp_path / "t1.jsonl"
    trace2 = tmp_path / "t2.jsonl"
    for p, tid, mem in ((trace1, "same", "m1"), (trace2, "same", "m2")):
        p.write_text(
            json.dumps({
                "event": "memory_decision",
                "memory_id": mem,
                "controller_prompt": "test prompt",
                "controller_output": '{"visibility":"global","supersedes":null}',
                "proposal": {"task_id": tid, "agent_id": "a", "raw_value": "val"},
                "target": {"visibility": "global", "supersedes": None},
            }) + "\n",
            encoding="utf-8",
        )

    out_dir = str(tmp_path / "grpo_adapter")
    res = train_qwen_rl(
        [str(trace1), str(trace2)],
        out_dir,
        "dummy/model",
        rewards=[1.0, 0.0],
        epochs=1,
    )
    assert res == out_dir
    meta_path = tmp_path / "grpo_adapter" / "adapter_metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["training_method"] == "trajectory_grpo"
    assert meta["group_size"] == 4
    assert meta["kl_anchor"] is True
    assert meta["epochs"] == 1


def test_export_sft_pairs_serializes_targeted_recipients_as_list(tmp_path):
    trace = tmp_path / "targeted.jsonl"
    ev = {
        "event": "memory_decision",
        "memory_id": "m_target",
        "controller_prompt": "targeted prompt",
        "proposal": {
            "proposal_id": "p_target",
            "task_id": "t",
            "agent_id": "worker_1",
            "title": "targeted note",
            "raw_value": "secret",
            "step_index": 1,
        },
        "target": {
            "visibility": "targeted",
            "target_recipients": ["worker_2"],
            "supersedes": None,
        },
    }
    trace.write_text(json.dumps(ev) + "\n", encoding="utf-8")
    pairs = export_sft_pairs([str(trace)])
    assert len(pairs) == 1
    prompt, completion = pairs[0]
    parsed = json.loads(completion)
    assert parsed["visibility"] == ["worker_2"]
    assert parsed["supersedes"] is None

    # Verify audit counts it under private
    report = audit_sft_distribution(pairs)
    assert report["counts"]["private"] == 1
    assert report["counts"]["invalid"] == 0


def test_load_qwen_rl_samples_reads_real_task_success_from_summary(tmp_path):
    t1_dir = tmp_path / "task1_r1"
    t2_dir = tmp_path / "task1_r2"
    t1_dir.mkdir()
    t2_dir.mkdir()

    trace1 = t1_dir / "memory_trace.jsonl"
    trace2 = t2_dir / "memory_trace.jsonl"
    summary1 = t1_dir / "summary.json"
    summary2 = t2_dir / "summary.json"

    ev1 = {
        "event": "memory_decision",
        "memory_id": "m1",
        "controller_prompt": "prompt 1",
        "controller_output": '{"visibility": "global", "supersedes": null}',
        "proposal": {"proposal_id": "p1", "task_id": "task_A", "agent_id": "a1", "raw_value": "data"},
        "target": {"visibility": "global", "supersedes": None},
    }
    ev2 = {
        "event": "memory_decision",
        "memory_id": "m2",
        "controller_prompt": "prompt 2",
        "controller_output": '{"visibility": "global", "supersedes": null}',
        "proposal": {"proposal_id": "p2", "task_id": "task_A", "agent_id": "a1", "raw_value": "data"},
        "target": {"visibility": "global", "supersedes": None},
    }
    trace1.write_text(json.dumps(ev1) + "\n", encoding="utf-8")
    trace2.write_text(json.dumps(ev2) + "\n", encoding="utf-8")

    # Score is 0.8, but task_success is True in summary
    summary1.write_text(json.dumps({"benchmark": "bm", "task_score": 0.8, "task_success": True}), encoding="utf-8")
    summary2.write_text(json.dumps({"benchmark": "bm", "task_score": 0.6, "task_success": False}), encoding="utf-8")

    samples, stats = load_qwen_rl_samples([str(trace1), str(trace2)], rewards=[0.8, 0.6])
    assert stats["samples"] == 2
    assert len(samples) == 2


