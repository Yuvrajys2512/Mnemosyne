from mnemosyne.types import EpisodicEvent, SemanticFact, EventType, FactType


def test_episodic_event_defaults():
    event = EpisodicEvent(
        content="User prefers Python",
        session_id="test",
        event_type=EventType.USER_MESSAGE,
    )
    assert event.id.startswith("ep_")
    assert event.importance == 0.5
    assert event.consolidated is False
    assert event.access_count == 0
    assert event.timestamp > 0


def test_episodic_event_ids_are_unique():
    a = EpisodicEvent(content="a", session_id="s", event_type=EventType.USER_MESSAGE)
    b = EpisodicEvent(content="b", session_id="s", event_type=EventType.USER_MESSAGE)
    assert a.id != b.id


def test_semantic_fact_defaults():
    fact = SemanticFact(
        content="User prefers Python",
        session_id="test",
        fact_type=FactType.USER_PREFERENCE,
    )
    assert fact.id.startswith("sem_")
    assert fact.importance == 0.5
    assert fact.confidence == 0.8
    assert fact.contradicted_by is None
    assert fact.supersedes is None


def test_semantic_fact_ids_are_unique():
    a = SemanticFact(content="a", session_id="s", fact_type=FactType.USER_PREFERENCE)
    b = SemanticFact(content="b", session_id="s", fact_type=FactType.USER_PREFERENCE)
    assert a.id != b.id
