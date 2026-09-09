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
    stats = EpisodeStats(task_score=1.0, read_tokens=512,
                         active_global_tokens=1024, token_budget=1024)
    # memory_cost = (512 + 1024 + 0)/1024 = 1.5 ; R = 1.0 - 0.05*1.5
    assert abs(episode_reward(stats) - (1.0 - 0.05 * 1.5)) < 1e-9


def test_episode_reward_communication_cost():
    stats = EpisodeStats(task_score=1.0, communication_tokens=256, token_budget=1024)
    # memory_cost = 256/1024 = 0.25 ; R = 1.0 - 0.05*0.25
    assert abs(episode_reward(stats) - (1.0 - 0.05 * 0.25)) < 1e-9
    # zero cost leaves task score untouched
    assert episode_reward(EpisodeStats(task_score=1.0)) == 1.0


def _proposal(pid, agent, value, topics=()):
    return MemoryProposal(proposal_id=pid, task_id="t", agent_id=agent,
                          source="worker", title=value[:8], raw_value=value,
                          step_index=1, topics=topics)


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
    credits = proposal_rewards(events, task_score=1.0)
    assert credits[m2.memory_id] < 0  # unread: pure storage cost
    # read by non-owner: A_e=1.0, mult=1+0.25, minus lambda*cost
    expected = 1.0 * (1 + 0.25) - 0.05 * token_count("alpha beta gamma") / 4096
    assert abs(credits[m1.memory_id] - expected) < 1e-9
    assert credits[m1.memory_id] > credits[m2.memory_id]


def test_proposal_rewards_owner_read_zero_advantage(tmp_path):
    mem = GovernedMemory(MemoryBank(), PrivateOnlyController())
    item = mem.submit(_proposal("p1", "coder", "only mine"))
    assert item is not None
    mem.read(item.memory_id, "coder", "t")
    credits = proposal_rewards([
        {"event": "memory_decision", "memory_id": item.memory_id,
         "proposal": {"agent_id": "coder", "raw_value": "only mine"}},
        {"event": "memory_read", "memory_id": item.memory_id,
         "reader_id": "coder", "task_id": "t"},
    ], task_score=0.5, same_task_baseline=0.5)
    # advantage 0 -> credit is just -lambda*cost
    expected = -0.05 * token_count("only mine") / 4096
    assert abs(credits[item.memory_id] - expected) < 1e-9


def test_proposal_rewards_rlvr_outcome_gating_blocks_failed_task(tmp_path):
    events, m1, m2 = _events(tmp_path)
    # Failed task (score 0.6 < 1.0) with positive advantage:
    # With outcome_gated=True, non-owner reads get mult=1.0 (no beta bonus)
    credits_gated = proposal_rewards(events, task_score=0.6, same_task_baseline=0.0, outcome_gated=True)
    expected_gated = 0.6 * 1.0 - 0.05 * token_count("alpha beta gamma") / 4096
    assert abs(credits_gated[m1.memory_id] - expected_gated) < 1e-9

    # Without outcome gating, proxy reward hacking occurs (mult = 1 + 0.25 = 1.25)
    credits_ungated = proposal_rewards(events, task_score=0.6, same_task_baseline=0.0, outcome_gated=False)
    expected_ungated = 0.6 * 1.25 - 0.05 * token_count("alpha beta gamma") / 4096
    assert abs(credits_ungated[m1.memory_id] - expected_ungated) < 1e-9
    assert credits_ungated[m1.memory_id] > credits_gated[m1.memory_id]


def test_proposal_rewards_simpo_density_penalty():
    # 20 global memory decisions exceeding target_card_budget=10
    events = []
    for i in range(20):
        events.append({
            "event": "memory_decision",
            "memory_id": f"m{i}",
            "target": {"visibility": "global"},
            "proposal": {"agent_id": "a1", "raw_value": "test card"},
        })
    credits = proposal_rewards(events, task_score=1.0, same_task_baseline=0.0,
                               target_card_budget=10, gamma_density=0.04)
    # unread cards incur -lambda*cost - density_penalty
    # density_penalty = 0.04 * (20 - 10) / 10 = 0.04
    cost = token_count("test card") / 4096
    expected = -0.05 * cost - 0.04
    assert abs(credits["m0"] - expected) < 1e-9


def test_proposal_rewards_includes_absent_decisions():
    # Chapter 6: Absent proposals receive credit = 0.0 (no free advantage for inaction) and are keyed by proposal_id
    events = [
        {
            "event": "memory_decision",
            "memory_id": None,
            "target": {"visibility": "absent"},
            "proposal": {"proposal_id": "prop_absent_1", "agent_id": "a1", "raw_value": "noise"},
        },
        {
            "event": "memory_decision",
            "memory_id": "m1",
            "target": {"visibility": "private"},
            "proposal": {"proposal_id": "prop_stored_1", "agent_id": "a1", "raw_value": "useful"},
        },
    ]
    credits = proposal_rewards(events, task_score=0.9, same_task_baseline=0.4)
    # Advantage = 0.5, but absent proposal receives credit = 0.0
    assert credits["prop_absent_1"] == 0.0
    assert "prop_stored_1" in credits
    assert "m1" in credits


def test_proposal_rewards_harmful_cross_read_penalized():
    # Harmful cross-agent read (A < 0) incurs harmful_penalty (default 0.2)
    events = [
        {
            "event": "memory_decision",
            "memory_id": "m_harmful",
            "target": {"visibility": "global"},
            "proposal": {"proposal_id": "p_harm", "agent_id": "a1", "raw_value": "bad advice"},
        },
        {
            "event": "memory_read",
            "memory_id": "m_harmful",
            "reader_id": "a2",
        },
        {
            "event": "memory_proposal",
            "proposal": {"agent_id": "a2", "raw_value": "confused action"},
        },
    ]
    # Negative advantage: task_score = -0.5, baseline = 0.0 -> A = -0.5
    credits = proposal_rewards(events, task_score=-0.5, same_task_baseline=0.0)
    cost = token_count("bad advice") / 4096
    expected = -0.5 - 0.05 * cost - 0.2  # A - lam*cost - harmful_penalty
    assert abs(credits["m_harmful"] - expected) < 1e-9



def test_targeted_memory_collaboration_and_density_exemption():
    """Targeted memory read by non-owner gets collaboration bonus and is exempt from density penalty."""
    events = [
        # 15 global cards that exceed the budget of 10
        *[{
            "event": "memory_decision",
            "memory_id": f"g_{i}",
            "target": {"visibility": "global"},
            "proposal": {"proposal_id": f"p_g_{i}", "agent_id": "a1", "raw_value": "global data"},
        } for i in range(15)],
        # 1 targeted card from a1 targeted to a2
        {
            "event": "memory_decision",
            "memory_id": "m_targeted",
            "target": {"visibility": "targeted", "target_recipients": ["a2"]},
            "proposal": {"proposal_id": "p_target", "agent_id": "a1", "raw_value": "targeted diagnostic hint"},
        },
        # a2 reads m_targeted
        {
            "event": "memory_read",
            "memory_id": "m_targeted",
            "reader_id": "a2",
        },
    ]

    credits = proposal_rewards(
        events,
        task_score=1.0,
        same_task_baseline=0.0,
        target_card_budget=10,
        gamma_density=0.10,
        task_success=True,
    )

    cost = token_count("targeted diagnostic hint") / 4096
    # Advantage = 1.0, mult = 1.0 + 0.25 = 1.25 (since read by non-owner a2)
    # Density penalty must NOT apply to targeted memory!
    # Target hit bonus (+0.3) applies because a2 was in target_recipients=["a2"]
    expected_targeted_credit = 1.0 * 1.25 + 0.3 - 0.05 * cost
    assert abs(credits["m_targeted"] - expected_targeted_credit) < 1e-9

    # Global cards DO suffer density penalty
    # density_penalty = 0.10 * (15 - 10) / 10 = 0.05
    global_cost = token_count("global data") / 4096
    expected_global_unread = -0.05 * global_cost - 0.05
    assert abs(credits["g_0"] - expected_global_unread) < 1e-9


def test_targeted_memory_hit_bonus_on_zero_advantage():
    """Targeted memory with target hit receives target_bonus even when advantage is zero."""
    events = [
        {
            "event": "memory_decision",
            "memory_id": "m_target",
            "target": {"visibility": "targeted", "target_recipients": ["worker_2"]},
            "proposal": {"proposal_id": "p_target", "agent_id": "worker_1", "raw_value": "database schema details"},
        },
        {
            "event": "memory_read",
            "memory_id": "m_target",
            "reader_id": "worker_2",
        },
    ]
    # Advantage 0.0, but successful task
    credits = proposal_rewards(events, task_score=0.8, same_task_baseline=0.8, task_success=True)
    cost = token_count("database schema details") / 4096
    # A + target_bonus - lam * cost = 0.0 + 0.3 - 0.05 * cost > 0
    expected = 0.0 + 0.3 - 0.05 * cost
    assert abs(credits["m_target"] - expected) < 1e-9
    assert credits["m_target"] > 0.0  # Strictly superior to absent (0.0)!


def test_successful_task_with_slight_negative_advantage_not_penalized_as_harmful():
    """Successful task with negative intra-group advantage is NOT hit with harmful_penalty."""
    events = [
        {
            "event": "memory_decision",
            "memory_id": "m_cross",
            "target": {"visibility": "global"},
            "proposal": {"proposal_id": "p_cross", "agent_id": "a1", "raw_value": "shared finding"},
        },
        {
            "event": "memory_read",
            "memory_id": "m_cross",
            "reader_id": "a2",
        },
        {
            "event": "memory_decision",
            "memory_id": "m_a2",
            "target": {"visibility": "global"},
            "proposal": {"proposal_id": "p_a2", "agent_id": "a2", "raw_value": "downstream act"},
        },
    ]
    # Advantage is slightly negative (e.g. -0.2), but task succeeded (task_success=True)
    credits = proposal_rewards(
        events,
        task_score=0.8,
        same_task_baseline=0.85,
        advantage=-0.2,
        task_success=True,
    )
    cost = token_count("shared finding") / 4096
    # Must NOT suffer harmful_penalty (-0.2): credit is A - lam * cost = -0.2 - 0.05 * cost
    expected = -0.2 - 0.05 * cost
    assert abs(credits["m_cross"] - expected) < 1e-9



