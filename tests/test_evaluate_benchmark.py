import json

from marble.experiments.evaluate import (
    aggregate_by_method,
    compute_paired_memory_dependency,
    evaluate_memory_trace,
    evaluate_run_root,
    evaluate_task_dir,
    main,
)


def _decision(mid, agent, vis, supersedes=None, value="a b c"):
    return {
        "event": "memory_decision",
        "memory_id": mid,
        "proposal": {"agent_id": agent, "raw_value": value},
        "target": {"visibility": vis, "supersedes": supersedes},
    }


def _read(mid, reader):
    return {"event": "memory_read", "memory_id": mid, "reader_id": reader}


def test_trace_metrics_core():
    # Empty trace defaults
    m0 = evaluate_memory_trace([])
    assert m0["decisions_total"] == 0 and m0["reads"] == 0
    assert m0["accept_rate"] == 0.0 and m0["reuse_rate"] == 0.0

    # Decision counts and basic metrics
    events = [
        {"event": "memory_decision"},  # rejected: no memory_id
        _decision("m1", "a1", "global"),
        _decision("m2", "a2", "private"),
        _decision("m3", "a1", "global", supersedes="m1"),
        _read("m1", "a2"),  # cross-agent
        _read("m2", "a2"),  # owner
    ]
    m = evaluate_memory_trace(events)
    assert m["decisions_total"] == 3
    assert abs(m["accept_rate"] - 0.75) < 1e-9
    assert abs(m["reject_rate"] - 0.25) < 1e-9
    assert m["private"] == 1 and m["global"] == 2
    assert m["supersessions"] == 1
    assert m["reads"] == 2 and m["cross_agent_reads"] == 1
    assert abs(m["reuse_rate"] - 2 / 3) < 1e-9
    assert m["active_global_tokens"] == 3
    assert m["active_private_tokens"] == 3
    assert m["active_memory_count"] == 2

    # R1 CRUD events
    r1_events = [
        {"event": "memory_r1_operation", "operation": "ADD", "memory_id": "r1", "proposal": {"raw_value": "old result"}},
        {"event": "memory_r1_operation", "operation": "UPDATE", "memory_id": "r1", "proposal": {"raw_value": "corrected shared result"}},
        {"event": "memory_exposure", "memory_ids": ["r1"], "reader_id": "a2"},
        _read("r1", "a2"),
    ]
    m_r1 = evaluate_memory_trace(r1_events)
    assert (m_r1["r1_adds"], m_r1["r1_updates"], m_r1["r1_deletes"], m_r1["r1_noops"]) == (1, 1, 0, 0)
    assert m_r1["exposure_events"] == m_r1["exposed_cards"] == 1
    assert m_r1["active_global_tokens"] == 3


def test_evaluate_task_dir(tmp_path):
    tdir = tmp_path / "coding" / "5"
    tdir.mkdir(parents=True)
    summary = {
        "method": "heuristic", "benchmark": "coding", "task_id": 5,
        "seed": 1, "status": "ok", "task_score": 0.8,
        "task_success": 1.0, "score_status": "available",
        "agent_count": 4, "ablation": "input:no_task_goal",
        "manifest": "frozen.json", "setting": "method=heuristic|ablation=input:no_task_goal",
        "retrieval": {"name": "key_first", "max_cards": 3, "max_reads_per_step": 2},
        "episode_latency_s": 2.5, "api_calls": 6,
        "total_tokens": 120, "episode_reward": 0.7,
    }
    (tdir / "summary.json").write_text(json.dumps(summary))
    (tdir / "memory_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [_decision("m1", "a1", "global"), _read("m1", "a2")])
    )
    (tdir / "reward.json").write_text(json.dumps({"m1": 1.2}))

    row = evaluate_task_dir(tdir)
    assert row["method"] == "heuristic" and row["task_score"] == 0.8
    assert row["memory"]["reads"] == 1
    assert row["reward"] == 1.2
    assert row["agent_count"] == 4
    assert row["setting"] == summary["setting"]
    assert row["ablation"] == "input:no_task_goal"
    assert row["manifest"] == "frozen.json"
    assert row["retrieval"]["max_cards"] == 3


def test_evaluate_aggregation_and_cli(tmp_path, capsys):
    # Walk and aggregation
    for method, score, score_status in (
        ("heuristic", 0.8, "available"),
        ("heuristic", 0.6, "available"),
        ("heuristic", 100.0, "unavailable"),
        ("no_memory", 0.5, "available"),
    ):
        tdir = tmp_path / "coding" / method / str(score)
        tdir.mkdir(parents=True)
        (tdir / "summary.json").write_text(json.dumps(
            {"method": method, "benchmark": "coding", "task_id": 1,
             "seed": None, "status": "ok", "task_score": score,
             "score_status": score_status}))

    rows = evaluate_run_root(tmp_path)
    assert len(rows) == 3
    agg = aggregate_by_method(rows)
    assert abs(agg["heuristic"]["task_score"] - 0.7) < 1e-9
    assert agg["no_memory"]["task_score"] == 0.5

    # Manifest filter
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "splits": {
            "train": [{"benchmark": "coding", "task_id": 1}],
            "test": [],
        }
    }))
    assert len(evaluate_run_root(tmp_path, manifest=manifest, split="train")) >= 1

    # CLI evaluation
    main(["--run-dir", str(tmp_path), "--include-unavailable"])
    out = json.loads(capsys.readouterr().out)
    assert "heuristic" in out and "no_memory" in out


def test_read_metrics_and_dependency():
    # 1. Read coverage and repeated reuse
    events = [
        _decision("m1", "a1", "global"),
        _decision("m2", "a1", "global"),
        _read("m1", "a2"),
        _read("m1", "a3"),
        _read("m1", "a1"),
    ]
    m = evaluate_memory_trace(events)
    assert m["read_coverage"] == 0.5
    assert m["repeated_reuse_rate"] == 0.5
    assert m["cross_agent_read_coverage"] == 0.5

    # 2. Stored, valid absent, format error
    events_err = [
        _decision("m1", "a1", "global"),
        {"event": "memory_decision", "memory_id": None, "target": {"visibility": "absent"}, "proposal": {"proposal_id": "p2", "agent_id": "a1", "raw_value": "noise"}, "parse_status": "valid_json"},
        {"event": "memory_decision", "memory_id": None, "target": {"visibility": "absent"}, "proposal": {"proposal_id": "p3", "agent_id": "a1", "raw_value": "broken"}, "parse_status": "format_error"},
    ]
    m_err = evaluate_memory_trace(events_err)
    assert m_err["proposals_total"] == 3
    assert abs(m_err["stored_rate"] - 1 / 3) < 1e-9
    assert abs(m_err["format_error_rate"] - 1 / 3) < 1e-9

    # 3. Paired memory dependency
    rows = [
        {"benchmark": "db", "task_id": 1, "method": "global_add_all", "task_score": 1.0},
        {"benchmark": "db", "task_id": 1, "method": "no_memory", "task_score": 0.2},
        {"benchmark": "db", "task_id": 2, "method": "global_add_all", "task_score": 0.5},
        {"benchmark": "db", "task_id": 2, "method": "no_memory", "task_score": 0.5},
    ]
    res = compute_paired_memory_dependency(rows)
    assert res["paired_tasks_count"] == 2
    assert res["memory_sensitive_count"] == 1
    assert res["memory_insensitive_count"] == 1


def test_collaboration_and_transfer_metrics():
    # 1. Spec 12.7: exposure, reuse, negative transfer
    events_spec = [
        _decision("m_priv", "a1", "private"),
        _decision("m_glob", "a1", "global"),
        {"event": "memory_exposure", "memory_ids": ["m_priv"], "reader_id": "a1"},
        {"event": "memory_exposure", "memory_ids": ["m_glob"], "reader_id": "a2"},
        _read("m_priv", "a1"),
        _read("m_glob", "a2"),
    ]
    m_ok = evaluate_memory_trace(events_spec, task_success=1.0)
    assert m_ok["cross_agent_reads"] == 1
    assert m_ok["negative_transfer"] == 0
    m_fail = evaluate_memory_trace(events_spec, task_success=0.0)
    assert m_fail["negative_transfer"] == 1

    # 2. Targeted memory metrics
    events_targ = [
        {"event": "memory_decision", "memory_id": "m_targ1", "proposal": {"agent_id": "a1", "raw_value": "plan"}, "target": {"visibility": "targeted", "target_recipients": ["a2"], "supersedes": None}},
        {"event": "memory_decision", "memory_id": "m_targ2", "proposal": {"agent_id": "a1", "raw_value": "arch"}, "target": {"visibility": "targeted", "target_recipients": ["a1", "a2"], "supersedes": None}},
        _read("m_targ1", "a2"),
        _read("m_targ2", "a1"),
    ]
    metrics_t = evaluate_memory_trace(events_targ, task_success=1.0)
    assert metrics_t["targeted_written"] == 2
    assert metrics_t["targeted_cross_read"] == 1

    # 3. Productive cross reads vs negative transfer
    events_active = [
        _decision("m1", "a1", "global"),
        _read("m1", "a2"),
        _decision("m2", "a2", "global"),
    ]
    m_pos = evaluate_memory_trace(events_active, task_success=1.0, advantage=0.8)
    assert m_pos["productive_cross_reads"] == 1
    assert m_pos["productive_cross_read_rate"] == 1.0

    m_neg = evaluate_memory_trace(events_active, task_success=0.0, advantage=-0.5)
    assert m_neg["harmful_cross_reads"] == 1
    assert m_neg["negative_transfer"] == 1.0
