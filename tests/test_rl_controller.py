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
