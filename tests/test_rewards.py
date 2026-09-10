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


def test_episode_reward_formula_and_communication():
    # Read tokens & global active tokens cost
    stats = EpisodeStats(task_score=1.0, read_tokens=512, active_global_tokens=1024, token_budget=1024)
    assert abs(episode_reward(stats) - (1.0 - 0.05 * 1.5)) < 1e-9

    # Communication cost
    stats_comm = EpisodeStats(task_score=1.0, communication_tokens=256, token_budget=1024)
    assert abs(episode_reward(stats_comm) - (1.0 - 0.05 * 0.25)) < 1e-9
    assert episode_reward(EpisodeStats(task_score=1.0)) == 1.0


def _proposal(pid, agent, value, topics=()):
    return MemoryProposal(proposal_id=pid, task_id="t", agent_id=agent,
                          source="worker", title=value[:8], raw_value=value,
                          step_index=1, topics=topics)


def _events(tmp_path):
    trace_path = str(tmp_path / "trace.jsonl")
    mem = GovernedMemory(MemoryBank(), GlobalAlwaysController(), trace=TraceLogger(trace_path))
    m1 = mem.submit(_proposal("p1", "coder", "alpha beta gamma"))
    m2 = mem.submit(_proposal("p2", "reviewer", "delta"))
    assert m1 is not None and m2 is not None
    mem.read(m1.memory_id, "reviewer", "t")
    with open(trace_path) as fh:
        events = [json.loads(line) for line in fh]
    return events, m1, m2


def test_proposal_rewards_reads_and_outcome_gating(tmp_path):
    events, m1, m2 = _events(tmp_path)
    credits = proposal_rewards(events, task_score=1.0)
    assert credits[m2.memory_id] < 0
    expected = 1.0 * (1 + 0.25) - 0.05 * token_count("alpha beta gamma") / 4096
    assert abs(credits[m1.memory_id] - expected) < 1e-9
    assert credits[m1.memory_id] > credits[m2.memory_id]

    # Owner read with zero advantage
    mem = GovernedMemory(MemoryBank(), PrivateOnlyController())
    item = mem.submit(_proposal("p_own", "coder", "only mine"))
    mem.read(item.memory_id, "coder", "t")
    credits_own = proposal_rewards([
        {"event": "memory_decision", "memory_id": item.memory_id, "proposal": {"agent_id": "coder", "raw_value": "only mine"}},
        {"event": "memory_read", "memory_id": item.memory_id, "reader_id": "coder", "task_id": "t"},
    ], task_score=0.5, same_task_baseline=0.5)
    assert abs(credits_own[item.memory_id] - (-0.05 * token_count("only mine") / 4096)) < 1e-9

    # RLVR outcome gating
    credits_gated = proposal_rewards(events, task_score=0.6, same_task_baseline=0.0, outcome_gated=True)
    expected_gated = 0.6 * 1.0 - 0.05 * token_count("alpha beta gamma") / 4096
    assert abs(credits_gated[m1.memory_id] - expected_gated) < 1e-9
    credits_ungated = proposal_rewards(events, task_score=0.6, same_task_baseline=0.0, outcome_gated=False)
    assert credits_ungated[m1.memory_id] > credits_gated[m1.memory_id]


def test_density_penalty_and_absent_decisions():
    # 20 global memory decisions exceeding budget of 10
    events = [{
        "event": "memory_decision",
        "memory_id": f"m{i}",
        "target": {"visibility": "global"},
        "proposal": {"agent_id": "a1", "raw_value": "test card"},
    } for i in range(20)]
    credits = proposal_rewards(events, task_score=1.0, same_task_baseline=0.0, target_card_budget=10, gamma_density=0.04)
    expected = -0.05 * (token_count("test card") / 4096) - 0.04
    assert abs(credits["m0"] - expected) < 1e-9

    # Absent decisions receiving credit = 0.0
    events_abs = [
        {"event": "memory_decision", "memory_id": None, "target": {"visibility": "absent"}, "proposal": {"proposal_id": "prop_absent_1", "agent_id": "a1", "raw_value": "noise"}},
        {"event": "memory_decision", "memory_id": "m1", "target": {"visibility": "private"}, "proposal": {"proposal_id": "prop_stored_1", "agent_id": "a1", "raw_value": "useful"}},
    ]
    credits_abs = proposal_rewards(events_abs, task_score=0.9, same_task_baseline=0.4)
    assert credits_abs["prop_absent_1"] == 0.0
    assert "prop_stored_1" in credits_abs


def test_targeted_memory_and_harmful_penalties():
    # Harmful cross-read penalized on failed task
    events_harm = [
        {"event": "memory_decision", "memory_id": "m_harmful", "target": {"visibility": "global"}, "proposal": {"proposal_id": "p_harm", "agent_id": "a1", "raw_value": "bad advice"}},
        {"event": "memory_read", "memory_id": "m_harmful", "reader_id": "a2"},
        {"event": "memory_proposal", "proposal": {"agent_id": "a2", "raw_value": "confused action"}},
    ]
    credits_harm = proposal_rewards(events_harm, task_score=-0.5, same_task_baseline=0.0)
    expected_harm = -0.5 - 0.05 * (token_count("bad advice") / 4096) - 0.2
    assert abs(credits_harm["m_harmful"] - expected_harm) < 1e-9

    # Targeted memory collaboration & density exemption
    events_targ = [
        *[{
            "event": "memory_decision", "memory_id": f"g_{i}", "target": {"visibility": "global"},
            "proposal": {"proposal_id": f"p_g_{i}", "agent_id": "a1", "raw_value": "global data"},
        } for i in range(15)],
        {"event": "memory_decision", "memory_id": "m_targeted", "target": {"visibility": "targeted", "target_recipients": ["a2"]}, "proposal": {"proposal_id": "p_target", "agent_id": "a1", "raw_value": "hint"}},
        {"event": "memory_read", "memory_id": "m_targeted", "reader_id": "a2"},
    ]
    credits_targ = proposal_rewards(events_targ, task_score=1.0, same_task_baseline=0.0, target_card_budget=10, gamma_density=0.10, task_success=True)
    cost = token_count("hint") / 4096
    assert abs(credits_targ["m_targeted"] - (1.0 * 1.25 + 0.3 - 0.05 * cost)) < 1e-9
    assert credits_targ["g_0"] < 0

    # Targeted memory hit bonus on zero advantage
    events_zero = [
        {"event": "memory_decision", "memory_id": "m_target", "target": {"visibility": "targeted", "target_recipients": ["worker_2"]}, "proposal": {"proposal_id": "p_target", "agent_id": "worker_1", "raw_value": "details"}},
        {"event": "memory_read", "memory_id": "m_target", "reader_id": "worker_2"},
    ]
    credits_zero = proposal_rewards(events_zero, task_score=0.8, same_task_baseline=0.8, task_success=True)
    assert credits_zero["m_target"] > 0.0

    # Successful task with slight negative advantage is NOT hit with harmful penalty
    events_succ_neg = [
        {"event": "memory_decision", "memory_id": "m_cross", "target": {"visibility": "global"}, "proposal": {"proposal_id": "p_cross", "agent_id": "a1", "raw_value": "finding"}},
        {"event": "memory_read", "memory_id": "m_cross", "reader_id": "a2"},
        {"event": "memory_decision", "memory_id": "m_a2", "target": {"visibility": "global"}, "proposal": {"proposal_id": "p_a2", "agent_id": "a2", "raw_value": "act"}},
    ]
    credits_succ_neg = proposal_rewards(events_succ_neg, task_score=0.8, same_task_baseline=0.85, advantage=-0.2, task_success=True)
    expected_succ = -0.2 - 0.05 * (token_count("finding") / 4096)
    assert abs(credits_succ_neg["m_cross"] - expected_succ) < 1e-9
