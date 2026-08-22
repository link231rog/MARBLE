import json

from marble.controllers import GlobalAlwaysController, PrivateOnlyController
from marble.memory import (
    EpisodeStats,
    GovernedMemory,
    MemoryBank,
    MemoryProposal,
    TraceLogger,
    episode_reward,
    proposal_rewards,
    token_count,
)


def test_episode_reward_formula():
    stats = EpisodeStats(task_score=1.0, coordination_score=0.5,
                         read_tokens=512, active_global_tokens=1024,
                         token_budget=1024)
    r = episode_reward(stats)
    assert abs(r - (1.0 + 0.25 * 0.5 - 0.10 * 0.5 - 0.05 * 1.0)) < 1e-9


def test_episode_reward_communication_cost():
    stats = EpisodeStats(task_score=1.0, communication_tokens=256, token_budget=1024)
    assert abs(episode_reward(stats) - (1.0 - 0.10 * 0.25)) < 1e-9
    # default zero keeps old formula
    assert episode_reward(EpisodeStats(task_score=1.0)) == 1.0


def _proposal(pid, agent, value):
    return MemoryProposal(proposal_id=pid, task_id="t", agent_id=agent,
                          source="worker", title=value[:8], raw_value=value,
                          step_index=1)


def _events(tmp_path):
    trace_path = str(tmp_path / "trace.jsonl")
    mem = GovernedMemory(MemoryBank(), GlobalAlwaysController(),
                         trace=TraceLogger(trace_path))
    m1 = mem.submit(_proposal("p1", "coder", "alpha beta gamma"))
    m2 = mem.submit(_proposal("p2", "reviewer", "delta"))
    assert m1 is not None and m2 is not None
    mem.read(m1.memory_id, "reviewer", "t")  # non-owner read
    with open(trace_path) as fh:
        events = [json.loads(line) for line in fh]
    return events, m1, m2


def test_proposal_rewards_non_owner_read_weighted(tmp_path):
    events, m1, m2 = _events(tmp_path)
    credits = proposal_rewards(events, r_episode=1.0)
    assert credits[m2.memory_id] < 0  # unread: pure storage cost
    # read by non-owner: -storage + 1.25 * (1.0 - 0)
    assert credits[m1.memory_id] > credits[m2.memory_id]
    assert abs(credits[m1.memory_id] - (-3 / 4096 + 1.25)) < 1e-9


def test_proposal_rewards_owner_read_baseline(tmp_path):
    mem = GovernedMemory(MemoryBank(), PrivateOnlyController())
    item = mem.submit(_proposal("p1", "coder", "only mine"))
    assert item is not None
    mem.read(item.memory_id, "coder", "t")
    credits = proposal_rewards([
        {"event": "memory_decision", "memory_id": item.memory_id,
         "proposal": {"agent_id": "coder", "raw_value": "only mine"}},
        {"event": "memory_read", "memory_id": item.memory_id,
         "reader_id": "coder", "task_id": "t"},
    ], r_episode=0.5, running_baseline=0.5)
    expected = -token_count("only mine") / 4096
    assert abs(credits[item.memory_id] - expected) < 1e-9
