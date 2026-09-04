from marble.engine.communication_governor import CommunicationGovernor


def test_communication_governor_basic():
    gov = CommunicationGovernor(echo_similarity_threshold=0.6)
    cards = [
        {"id": "M1", "key": "query pg stat statements for insert large data empty", "title": "Query INSERTs"}
    ]

    # Message that repeats M1 extensively
    verbose_msg = (
        "Hello agent2, I just queried the pg stat statements table for insert large data and found that it was completely empty. "
        "Therefore, I suggest we focus on investigating lock contention instead."
    )

    filtered, was_compressed = gov.filter_message("agent1", "agent2", verbose_msg, cards)
    assert was_compressed is True
    assert "[Governor Ref: M1" in filtered
    assert "investigating lock contention" in filtered

    # Message that has no relation to cards
    novel_msg = "Can someone check table permissions and user roles?"
    filtered2, was_compressed2 = gov.filter_message("agent1", "agent2", novel_msg, cards)
    assert was_compressed2 is False
    assert filtered2 == novel_msg


def test_communication_governor_metrics():
    gov = CommunicationGovernor()
    metrics = gov.get_efficiency_metrics()
    assert metrics["total_messages"] == 0
    assert metrics["compression_rate"] == 0.0
