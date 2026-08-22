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
    assert m["active_global_tokens"] == 6  # two global decisions × 3 words


def test_empty_trace_defaults():
    m = evaluate_memory_trace([])
    assert m["decisions_total"] == 0 and m["reads"] == 0
    assert m["accept_rate"] == 0.0 and m["reuse_rate"] == 0.0


def test_task_dir_join(tmp_path):
    tdir = tmp_path / "coding" / "5"
    tdir.mkdir(parents=True)
    (tdir / "summary.json").write_text(json.dumps(
        {"method": "heuristic", "benchmark": "coding", "task_id": 5,
         "seed": 1, "status": "ok", "task_score": 0.8}))
    (tdir / "memory_trace.jsonl").write_text("\n".join(
        json.dumps(e) for e in [_decision("m1", "a1", "global"), _read("m1", "a2")]))
    (tdir / "reward.json").write_text(json.dumps({"m1": 1.2}))
    row = evaluate_task_dir(tdir)
    assert row["method"] == "heuristic" and row["task_score"] == 0.8
    assert row["memory"]["reads"] == 1
    assert row["reward"] == 1.2


def test_run_root_walk_and_aggregation(tmp_path):
    for method, score in (("heuristic", 0.8), ("heuristic", 0.6), ("no_memory", 0.5)):
        tdir = tmp_path / "coding" / method / str(score)
        tdir.mkdir(parents=True)
        (tdir / "summary.json").write_text(json.dumps(
            {"method": method, "benchmark": "coding", "task_id": 1,
             "seed": None, "status": "ok", "task_score": score}))
    rows = evaluate_run_root(tmp_path)
    assert len(rows) == 3
    agg = aggregate_by_method(rows)
    assert abs(agg["heuristic"]["task_score"] - 0.7) < 1e-9
    assert agg["no_memory"]["task_score"] == 0.5
    assert agg["heuristic"]["task_score_se"] > 0
    assert agg["no_memory"]["task_score_se"] == 0.0


def test_cli_detects_deep_run_benchmark_layout(tmp_path, capsys):
    # layout produced by run_benchmark: root/run_id/baseline/benchmark/task_id/
    for baseline in ("no_memory", "heuristic"):
        tdir = tmp_path / "coding_multi" / baseline / "coding" / "1"
        tdir.mkdir(parents=True)
        (tdir / "summary.json").write_text(json.dumps(
            {"method": baseline, "benchmark": "coding", "task_id": 1,
             "seed": None, "status": "dry_run", "task_score": 0.0}))
        (tdir / "memory_trace.jsonl").write_text("")
    main(["--run-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert sorted(out) == ["heuristic", "no_memory"]
    assert out["heuristic"]["memory.decisions_total"] == 0.0
