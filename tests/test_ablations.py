import json

import pytest

from marble.controllers import GlobalAlwaysController, JsonController
from marble.experiments.ablations import (
    NoSupersede,
    apply_to_task_config,
    controller_kwargs,
    parse_ablation,
    reward_override,
    wrap_controller,
)
from marble.memory import GovernedMemory, MemoryBank, MemoryProposal


def _proposal(title="shared result", value="v"):
    return MemoryProposal(proposal_id="p1", task_id="t", agent_id="a1", source="worker",
                          title=title, raw_value=value, step_index=1)


def test_parse_ablation():
    assert parse_ablation("state_update:off") == ("state_update", "off")
    with pytest.raises(ValueError):
        parse_ablation("bogus:x")
    with pytest.raises(ValueError):
        parse_ablation("policy")


def test_schema_field_drop_removes_line():
    seen = {}

    def llm(prompt):
        seen["p"] = prompt
        return '{"visibility": "global", "supersedes": null}'

    ctrl = JsonController(llm, drop_fields=("value", "agent_id"))
    ctrl.decide(_proposal(), [])
    p = seen["p"]
    assert "value:" not in p and "agent_id:" not in p
    assert "title:" in p and "source:" in p
    assert controller_kwargs("schema_field", "value") == {"drop_fields": ("value",)}
    with pytest.raises(ValueError):
        controller_kwargs("schema_field", "bogus")


def test_state_update_off_forces_no_supersede():
    bank = MemoryBank()
    mem = GovernedMemory(bank, GlobalAlwaysController())
    first = mem.submit(_proposal())
    assert first is not None
    inner = GlobalAlwaysController()
    wrapped = wrap_controller(inner, "state_update", "off")
    assert isinstance(wrapped, NoSupersede)
    items = [it for it in bank.all_items() if it.active]
    target = wrapped.decide(_proposal(), items)
    assert target.supersedes is None
    # unwrapped would supersede (same title matches)
    assert inner.decide(_proposal(), items).supersedes == first.memory_id
    # no-op wrapper for other factors
    assert wrap_controller(inner, "retrieval", "none") is inner


def test_task_config_ablation_marks_memory_block():
    cfg = {"memory": {"backend": "governed", "controller": "heuristic",
                      "max_cards": 6, "max_reads_per_step": 2}}
    out = apply_to_task_config(cfg, "retrieval", "none")
    assert out["memory"]["max_cards"] == 0
    assert out["memory"]["ablation"] == "retrieval:none"
    pol = apply_to_task_config(cfg, "policy", "global_always")
    assert pol["memory"]["controller"] == "global_always"
    assert cfg["memory"]["max_cards"] == 6  # input untouched


def test_reward_override_zeroes_reuse(tmp_path):
    from marble.memory.rewards import proposal_rewards

    events = [
        {"event": "memory_decision", "memory_id": "m1",
         "proposal": {"agent_id": "a1", "raw_value": "x y"}},
        {"event": "memory_read", "memory_id": "m1", "reader_id": "a2"},
    ]
    base = proposal_rewards(events, r_episode=1.0)
    ablated = proposal_rewards(events, r_episode=1.0, **reward_override("reward", "no_reuse"))
    assert base["m1"] > 0 > ablated["m1"]  # reuse credit removed, storage cost remains
    assert reward_override("training", "scratch") == {}
