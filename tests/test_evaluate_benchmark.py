import json

from marble.experiments.evaluate import (
    aggregate_by_method,
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


def test_trace_metrics_counts():
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
    assert abs(m["accept_rate"] - 0.75) < 1e-9  # 3 stored / 4 decided
    assert abs(m["reject_rate"] - 0.25) < 1e-9
    assert m["private"] == 1 and m["global"] == 2
    assert m["supersessions"] == 1
    assert m["reads"] == 2 and m["cross_agent_reads"] == 1
    assert abs(m["reuse_rate"] - 2 / 3) < 1e-9
    assert m["active_global_tokens"] == 3  # superseded m1 is inactive
    assert m["active_private_tokens"] == 3
    assert m["active_memory_count"] == 2


def test_trace_metrics_join_r1_crud_and_exposure_events():
    events = [
        {
            "event": "memory_r1_operation",
            "operation": "ADD",
            "memory_id": "r1",
            "proposal": {"raw_value": "old result"},
        },
        {
            "event": "memory_r1_operation",
            "operation": "UPDATE",
            "memory_id": "r1",
            "proposal": {"raw_value": "corrected shared result"},
        },
        {"event": "memory_exposure", "memory_ids": ["r1"], "reader_id": "a2"},
        _read("r1", "a2"),
    ]

    m = evaluate_memory_trace(events)

    assert (m["r1_adds"], m["r1_updates"], m["r1_deletes"], m["r1_noops"]) == (
        1,
        1,
        0,
        0,
    )
    assert m["exposure_events"] == m["exposed_cards"] == 1
    assert m["active_global_tokens"] == 3


def test_empty_trace_defaults():
    m = evaluate_memory_trace([])
    assert m["decisions_total"] == 0 and m["reads"] == 0
    assert m["accept_rate"] == 0.0 and m["reuse_rate"] == 0.0


def test_task_dir_join(tmp_path):
    tdir = tmp_path / "coding" / "5"
    tdir.mkdir(parents=True)
    (tdir / "summary.json").write_text(json.dumps(
        {"method": "heuristic", "benchmark": "coding", "task_id": 5,
         "seed": 1, "status": "ok", "task_score": 0.8,
         "task_success": 1.0, "score_status": "available",
         "episode_latency_s": 2.5, "api_calls": 6,
         "total_tokens": 120, "episode_reward": 0.7}))
    (tdir / "memory_trace.jsonl").write_text("\n".join(
        json.dumps(e) for e in [_decision("m1", "a1", "global"), _read("m1", "a2")]))
    (tdir / "reward.json").write_text(json.dumps({"m1": 1.2}))
    row = evaluate_task_dir(tdir)
    assert row["method"] == "heuristic" and row["task_score"] == 0.8
    assert row["memory"]["reads"] == 1
    assert row["reward"] == 1.2
    assert row["task_success"] == 1.0
    assert row["api_calls"] == 6
    assert row["episode_reward"] == 0.7


def test_task_metadata_and_setting_are_preserved(tmp_path):
    tdir = tmp_path / "database" / "4"
    tdir.mkdir(parents=True)
    summary = {
        "method": "ours_rl", "benchmark": "database", "task_id": 4,
        "agent_count": 5, "seed": 42, "status": "ok",
        "ablation": "input:no_task_goal", "manifest": "frozen.json",
        "setting": "method=ours_rl|ablation=input:no_task_goal",
        "retrieval": {"name": "key_first", "max_cards": 3,
                      "max_reads_per_step": 2},
        "reward_config": {"lambda": 0.05, "beta": 0.25},
        "task_score": 0.9, "score_status": "available",
    }
    (tdir / "summary.json").write_text(json.dumps(summary))
    row = evaluate_task_dir(tdir)
    assert row["agent_count"] == 5
    assert row["setting"] == summary["setting"]
    assert row["ablation"] == "input:no_task_goal"
    assert row["manifest"] == "frozen.json"
    assert row["retrieval"]["max_cards"] == 3


def test_agent_count_metadata_supports_three_four_five_agents(tmp_path):
    for count in (3, 4, 5):
        tdir = tmp_path / str(count)
        tdir.mkdir()
        (tdir / "summary.json").write_text(json.dumps({
            "method": "ours_rl", "benchmark": "research", "task_id": count,
            "agent_count": count, "seed": 42, "setting": f"agents={count}",
            "status": "ok", "task_score": 0.5, "score_status": "available",
        }))
        row = evaluate_task_dir(tdir)
        assert row["agent_count"] == count
        assert row["setting"] == f"agents={count}"


def test_run_root_walk_and_aggregation(tmp_path):
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
    assert agg["heuristic"]["task_score_se"] > 0
    assert agg["no_memory"]["task_score_se"] == 0.0

    diagnostic_rows = evaluate_run_root(tmp_path, include_unavailable=True)
    assert len(diagnostic_rows) == 4
    diagnostic_agg = aggregate_by_method(
        diagnostic_rows,
        include_unavailable=True,
    )
    assert abs(diagnostic_agg["heuristic"]["task_score"] - 33.8) < 1e-9


def test_aggregation_keeps_settings_separate(tmp_path):
    for cards, score in ((1, 0.2), (6, 0.8)):
        tdir = tmp_path / "run" / "ours_rl" / "database" / str(cards)
        tdir.mkdir(parents=True)
        (tdir / "summary.json").write_text(json.dumps({
            "method": "ours_rl", "benchmark": "database", "task_id": cards,
            "agent_count": 4, "seed": 42, "status": "ok",
            "task_score": score, "score_status": "available",
            "setting": f"method=ours_rl|max_cards={cards}",
        }))
    rows = evaluate_run_root(tmp_path)
    agg = aggregate_by_method(rows)
    assert set(agg) == {"method=ours_rl|max_cards=1", "method=ours_rl|max_cards=6"}
    assert agg["method=ours_rl|max_cards=1"]["task_score"] == 0.2
    assert agg["method=ours_rl|max_cards=6"]["task_score"] == 0.8


def test_cli_detects_deep_run_benchmark_layout(tmp_path, capsys):
    # layout produced by run_benchmark: root/run_id/baseline/benchmark/task_id/
    for baseline in ("no_memory", "heuristic"):
        tdir = tmp_path / "coding_multi" / baseline / "coding" / "1"
        tdir.mkdir(parents=True)
        (tdir / "summary.json").write_text(json.dumps(
            {"method": baseline, "benchmark": "coding", "task_id": 1,
             "seed": None, "status": "dry_run", "task_score": 0.0,
             "score_status": "unavailable"}))
        (tdir / "memory_trace.jsonl").write_text("")
    main(["--run-dir", str(tmp_path), "--include-unavailable"])
    out = json.loads(capsys.readouterr().out)
    assert sorted(out) == ["heuristic", "no_memory"]
    assert out["heuristic"]["memory.decisions_total"] == 0.0
