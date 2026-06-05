"""
Tests for EpisodicStore.

We use a FakeEmbeddingFunction (deterministic hash-based vectors) so tests
run fast without downloading sentence-transformers models.
Semantic retrieval quality is tested separately (slow/integration tests).
"""
import hashlib
import time

import numpy as np
import pytest

from mnemosyne.core.episodic import EpisodicStore, _infer_importance
from mnemosyne.types import EventType


# ── Test fixture ──────────────────────────────────────────────────────────────

class FakeEmbeddingFunction:
    """Deterministic embeddings: same text → same vector, fast to compute."""

    def name(self) -> str:
        return "default"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            results.append(rng.standard_normal(384).tolist())
        return results

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        # Newer ChromaDB calls this for query-side embeddings separately
        return self._embed(input)


@pytest.fixture
def store(tmp_path):
    # tmp_path gives each test its own directory → no shared ChromaDB state
    return EpisodicStore(
        session_id="test_session",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
    )


# ── add() ─────────────────────────────────────────────────────────────────────

def test_add_returns_event_id(store):
    id_ = store.add("I prefer Python over JavaScript")
    assert id_.startswith("ep_")


def test_add_different_events_have_unique_ids(store):
    id_a = store.add("first message")
    id_b = store.add("second message")
    assert id_a != id_b


def test_add_trivial_content_is_skipped(store):
    id_ = store.add("ok")
    assert id_ == ""


def test_add_very_short_content_is_skipped(store):
    id_ = store.add("hi")
    assert id_ == ""


def test_add_content_over_2000_chars_is_truncated(store):
    long_content = "x" * 3000
    id_ = store.add(long_content)
    assert id_ != ""
    # Verify only truncated content stored
    events = store.get_recent(1)
    assert len(events[0].content) <= 2000


def test_add_custom_importance_is_respected(store):
    id_ = store.add("a message", importance=0.9)
    events = store.get_recent(1)
    assert events[0].importance == pytest.approx(0.9)


def test_add_metadata_is_stored(store):
    store.add("tool output", event_type=EventType.TOOL_RESULT, metadata={"tool_name": "search"})
    events = store.get_recent(1)
    assert events[0].metadata.get("tool_name") == "search"


# ── get_recent() ──────────────────────────────────────────────────────────────

def test_get_recent_returns_most_recent_first(store):
    store.add("first")
    time.sleep(0.01)
    store.add("second")
    time.sleep(0.01)
    store.add("third")

    recent = store.get_recent(3)
    assert recent[0].content == "third"
    assert recent[1].content == "second"
    assert recent[2].content == "first"


def test_get_recent_respects_n(store):
    for i in range(5):
        store.add(f"message {i}")
    assert len(store.get_recent(3)) == 3


def test_get_recent_empty_store_returns_empty(store):
    assert store.get_recent() == []


def test_get_recent_scoped_to_session(store, tmp_path):
    # A second store in the same directory but different session_id
    other = EpisodicStore(
        session_id="other_session",
        storage_path=str(tmp_path / "other"),
        _embedding_function=FakeEmbeddingFunction(),
    )
    store.add("my message")
    assert other.get_recent() == []


# ── query() ───────────────────────────────────────────────────────────────────

def test_query_returns_episodic_events(store):
    store.add("I prefer async Python patterns")
    store.add("The project uses FastAPI")
    results = store.query("Python preferences")
    assert len(results) > 0
    assert all(hasattr(r, "content") for r in results)


def test_query_empty_store_returns_empty(store):
    assert store.query("anything") == []


def test_query_respects_n_results(store):
    for i in range(10):
        store.add(f"unique message number {i} about something specific")
    results = store.query("message about something", n_results=3)
    assert len(results) <= 3


def test_query_with_metadata_filter(store):
    store.add("user said something", event_type=EventType.USER_MESSAGE)
    store.add("agent said something", event_type=EventType.AGENT_RESPONSE)

    results = store.query(
        "said something",
        where={"event_type": EventType.USER_MESSAGE.value},
    )
    assert all(r.event_type == EventType.USER_MESSAGE for r in results)


# ── consolidation ─────────────────────────────────────────────────────────────

def test_new_events_start_unconsolidated(store):
    store.add("first event")
    store.add("second event")
    assert store.count_unconsolidated() == 2


def test_mark_consolidated_updates_flag(store):
    id_ = store.add("an event")
    store.mark_consolidated([id_])

    events = store.get_unconsolidated()
    assert all(e.id != id_ for e in events)


def test_count_unconsolidated_decrements_after_mark(store):
    id_a = store.add("event a")
    id_b = store.add("event b")
    store.add("event c")

    assert store.count_unconsolidated() == 3
    store.mark_consolidated([id_a, id_b])
    assert store.count_unconsolidated() == 1


def test_get_unconsolidated_returns_oldest_first(store):
    store.add("old message")
    time.sleep(0.01)
    store.add("newer message")

    events = store.get_unconsolidated()
    assert events[0].timestamp <= events[-1].timestamp


def test_get_unconsolidated_respects_limit(store):
    for i in range(10):
        store.add(f"event {i}")
    events = store.get_unconsolidated(limit=3)
    assert len(events) <= 3


# ── importance inference ──────────────────────────────────────────────────────

def test_preference_keywords_boost_importance():
    score = _infer_importance("I always prefer Python", EventType.USER_MESSAGE)
    assert score > 0.5


def test_correction_keywords_boost_importance():
    score = _infer_importance("Actually that's incorrect", EventType.USER_MESSAGE)
    assert score > 0.5


def test_agent_response_lowers_importance():
    score = _infer_importance("Here is the answer you asked for", EventType.AGENT_RESPONSE)
    assert score < 0.5


def test_tool_result_lowers_importance():
    score = _infer_importance("Search results: blah blah blah", EventType.TOOL_RESULT)
    assert score < 0.5


def test_importance_is_clamped_to_0_1():
    # Multiple boosting keywords shouldn't exceed 1.0
    score = _infer_importance(
        "I must always prefer and never constraint requirement", EventType.USER_MESSAGE
    )
    assert 0.0 <= score <= 1.0
