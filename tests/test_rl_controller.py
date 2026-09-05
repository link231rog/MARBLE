import json

from marble.controllers.local_policy import LocalPolicyController
from marble.controllers.rl_controller import RLTrainer, train_rl


def _trace(tmp_path, name="trace.jsonl"):
    events = [
        {"event": "memory_decision", "memory_id": "m1",
         "proposal": {"title": "shared result", "raw_value": "a b", "source": "worker",
                      "agent_id": "a1", "task_id": "t", "step_index": 1},
         "target": {"visibility": "global"}},
        {"event": "memory_read", "memory_id": "m1", "reader_id": "a2"},
        {"event": "memory_decision", "memory_id": "m2",
         "proposal": {"title": "", "raw_value": "", "source": "worker",
                      "agent_id": "a1", "task_id": "t", "step_index": 2},
         "target": {"visibility": "absent"}},
    ]
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return str(p)


def test_positive_credit_reinforces_chosen():
    ctrl = LocalPolicyController()
    before = dict(ctrl.weights["global"])
    trainer = RLTrainer(ctrl, lr=0.1)
    assert trainer.update_event(
        {"proposal": {"title": "shared result", "raw_value": "x"}, "target": {"visibility": "global"}},
        credit=1.0,
    )
    after = ctrl.weights["global"]
    assert any(after[k] > before.get(k, 0.0) for k in after)


def test_negative_credit_pushes_away():
    ctrl = LocalPolicyController()
    trainer = RLTrainer(ctrl, lr=0.1)
    ev = {"proposal": {"title": "shared result", "raw_value": "x"},
          "target": {"visibility": "global"}}
    trainer.update_event(ev, credit=-1.0)
    # global score must not rise; absent/private gain on shared_signal feature
    assert all(w <= 0.0 for w in ctrl.weights["global"].values())


def test_invalid_visibility_skipped():
    trainer = RLTrainer(LocalPolicyController())
    assert not trainer.update_event({"proposal": {}, "target": {"visibility": "pending"}}, 1.0)


def test_train_rl_roundtrip(tmp_path):
    trace = _trace(tmp_path)
    out = tmp_path / "policy_rl.json"
    stats = train_rl([trace], str(out), epochs=2)
    assert stats["episodes"] == 1 and stats["updates"] > 0
    reloaded = LocalPolicyController.load(str(out))
    assert isinstance(reloaded, LocalPolicyController)


def test_train_rl_uses_real_task_scores(tmp_path):
    t1 = _trace(tmp_path, "t1.jsonl")
    t2 = _trace(tmp_path, "t2.jsonl")
    out = tmp_path / "policy_rl2.json"
    # real reward loop: per-episode task_score + split mean baseline
    stats = train_rl([t1, t2], str(out), epochs=2,
                     task_scores=[0.0, 1.0], same_task_baseline=0.5)
    assert stats["episodes"] == 2 and stats["updates"] > 0
    assert LocalPolicyController.load(str(out)) is not None


def test_train_rl_warmstart_and_anchor(tmp_path):
    init_ctrl = LocalPolicyController()
    init_ctrl.weights["global"]["shared_signal"] = 5.0
    init_path = tmp_path / "sft_init.json"
    init_ctrl.save(str(init_path))

    trace = _trace(tmp_path)
    out_path = tmp_path / "rl_anchored.json"
    stats = train_rl([trace], str(out_path), init_checkpoint=str(init_path),
                     epochs=3, anchor_coeff=0.1)
    assert stats["updates"] > 0
    loaded = LocalPolicyController.load(str(out_path))
    assert loaded.weights["global"]["shared_signal"] > 2.0


def test_rl_trainer_updates_absent_and_skips_format_error():
    ctrl = LocalPolicyController()
    ctrl.weights["global"]["bias"] = 1.0  # initial preference for global
    trainer = RLTrainer(ctrl, lr=0.1)
    events = [
        # Legitimate absent decision with positive credit (e.g. noise filtered during success)
        {
            "event": "memory_decision",
            "memory_id": None,
            "target": {"visibility": "absent"},
            "proposal": {"proposal_id": "p_absent", "title": "ignore", "raw_value": "noise"},
            "parse_status": "valid_json",
        },
        # Format error decision (must be skipped)
        {
            "event": "memory_decision",
            "memory_id": None,
            "target": {"visibility": "absent"},
            "proposal": {"proposal_id": "p_err", "title": "bad json", "raw_value": "bad"},
            "parse_status": "format_error",
        },
    ]
    credits = {"p_absent": 1.0, "p_err": 1.0}
    updates = trainer.update_trace(events, credits)
    assert updates == 1
    # Check that absent weights were reinforced
    assert ctrl.weights["absent"]["bias"] > 0.0


def test_rl_trainer_restores_active_state_for_features():
    ctrl = LocalPolicyController()
    trainer = RLTrainer(ctrl, lr=0.1)
    # Event with an active memory that matches the proposal title -> supersedes should be 1.0
    active_card = {
        "memory_id": "m_prior",
        "proposal_id": "p0",
        "task_id": "task_1",
        "title": "same plan",
        "raw_value": "old",
        "visibility": "global",
        "owner_id": None,
        "source_agent": "a1",
        "source": "worker",
        "step_index": 0,
        "active": True,
        "supersedes": None,
        "created_at": 100,
    }
    event = {
        "event": "memory_decision",
        "memory_id": "m1",
        "target": {"visibility": "global"},
        "proposal": {"proposal_id": "p1", "task_id": "task_1", "agent_id": "a1", "title": "same plan", "raw_value": "new"},
        "active_memory_index": [active_card],
    }
    assert trainer.update_event(event, credit=1.0) is True
    # has_supersedes weight for global should have increased from 0.0
    assert ctrl.weights["global"]["has_supersedes"] > 0.0


