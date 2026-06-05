"""
Tests for Consolidator — uses a MockProvider so no real LLM is called.
Tests cover: extraction parsing, batching, upsert integration, fallback logic.
"""
import hashlib
import json
import time

import numpy as np
import pytest

from mnemosyne import EventType, Mnemosyne
from mnemosyne.consolidator import (
    Consolidator,
    _batch_by_time,
    _format_events,
    _parse_response,
)
from mnemosyne.core.episodic import EpisodicStore
from mnemosyne.core.semantic import SemanticStore
from mnemosyne.providers import LLMProvider
from mnemosyne.types import EpisodicEvent, FactType, SemanticFact


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeEmbeddingFunction:
    def name(self):
        return "default"

    def _embed(self, texts):
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            results.append(rng.standard_normal(384).tolist())
        return results

    def __call__(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input)


class MockProvider(LLMProvider):
    """Returns a preset JSON response — no network calls."""

    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self):
        return "mock"

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps(self._response)


class FailingProvider(LLMProvider):
    @property
    def name(self):
        return "failing"

    def complete(self, system, user):
        raise RuntimeError("provider unavailable")


@pytest.fixture
def stores(tmp_path):
    ef = FakeEmbeddingFunction()
    episodic = EpisodicStore(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=ef,
    )
    semantic = SemanticStore(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=ef,
    )
    return episodic, semantic


# ── _parse_response ───────────────────────────────────────────────────────────

def test_parse_valid_response():
    raw = json.dumps({"facts": [
        {"content": "User prefers Python", "fact_type": "user_preference",
         "importance": 0.9, "confidence": 0.9},
    ]})
    facts = _parse_response(raw, session_id="test")
    assert len(facts) == 1
    assert facts[0].content == "User prefers Python"
    assert facts[0].fact_type == FactType.USER_PREFERENCE
    assert facts[0].importance == pytest.approx(0.9)


def test_parse_empty_facts_array():
    raw = json.dumps({"facts": []})
    assert _parse_response(raw, session_id="test") == []


def test_parse_invalid_json_returns_empty():
    facts = _parse_response("not json at all", session_id="test")
    assert facts == []


def test_parse_unknown_fact_type_falls_back_to_established_fact():
    raw = json.dumps({"facts": [
        {"content": "Something", "fact_type": "totally_unknown",
         "importance": 0.5, "confidence": 0.8},
    ]})
    facts = _parse_response(raw, session_id="test")
    assert facts[0].fact_type == FactType.ESTABLISHED_FACT


def test_parse_clamps_importance_to_0_1():
    raw = json.dumps({"facts": [
        {"content": "X", "fact_type": "user_preference", "importance": 5.0, "confidence": -1.0},
    ]})
    facts = _parse_response(raw, session_id="test")
    assert facts[0].importance == pytest.approx(1.0)
    assert facts[0].confidence == pytest.approx(0.0)


def test_parse_skips_empty_content():
    raw = json.dumps({"facts": [
        {"content": "", "fact_type": "user_preference", "importance": 0.5, "confidence": 0.5},
        {"content": "Valid fact", "fact_type": "user_preference", "importance": 0.5, "confidence": 0.5},
    ]})
    facts = _parse_response(raw, session_id="test")
    assert len(facts) == 1
    assert facts[0].content == "Valid fact"


# ── _batch_by_time ────────────────────────────────────────────────────────────

def test_batch_empty_returns_empty():
    assert _batch_by_time([], window_seconds=600) == []


def test_batch_single_event_returns_one_batch():
    e = EpisodicEvent(content="x", session_id="s", event_type=EventType.USER_MESSAGE)
    batches = _batch_by_time([e], window_seconds=600)
    assert len(batches) == 1
    assert batches[0] == [e]


def test_batch_close_events_are_grouped():
    t = time.time()
    events = [
        EpisodicEvent(content=f"msg {i}", session_id="s",
                      event_type=EventType.USER_MESSAGE, timestamp=t + i * 10)
        for i in range(5)
    ]
    batches = _batch_by_time(events, window_seconds=600)
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_batch_distant_events_are_split():
    t = time.time()
    early = EpisodicEvent(content="early", session_id="s",
                          event_type=EventType.USER_MESSAGE, timestamp=t)
    late = EpisodicEvent(content="late", session_id="s",
                         event_type=EventType.USER_MESSAGE, timestamp=t + 3600)
    batches = _batch_by_time([early, late], window_seconds=600)
    assert len(batches) == 2


# ── _format_events ────────────────────────────────────────────────────────────

def test_format_events_includes_content():
    e = EpisodicEvent(content="hello world", session_id="s",
                      event_type=EventType.USER_MESSAGE)
    formatted = _format_events([e])
    assert "hello world" in formatted


def test_format_events_numbers_lines():
    events = [
        EpisodicEvent(content=f"msg {i}", session_id="s",
                      event_type=EventType.USER_MESSAGE)
        for i in range(3)
    ]
    formatted = _format_events(events)
    assert "1." in formatted
    assert "2." in formatted
    assert "3." in formatted


# ── Consolidator integration ──────────────────────────────────────────────────

def test_consolidate_no_provider_returns_empty(stores):
    episodic, semantic = stores
    consolidator = Consolidator(
        session_id="test",
        episodic=episodic,
        semantic=semantic,
        provider=None,
    )
    result = consolidator.consolidate()
    assert result.episodes_processed == 0
    assert result.facts_created == 0


def test_consolidate_no_events_returns_empty(stores):
    episodic, semantic = stores
    consolidator = Consolidator(
        session_id="test",
        episodic=episodic,
        semantic=semantic,
        provider=MockProvider({"facts": []}),
    )
    result = consolidator.consolidate()
    assert result.episodes_processed == 0


def test_consolidate_extracts_and_stores_facts(stores):
    episodic, semantic = stores
    episodic.add("I prefer Python over JavaScript")
    episodic.add("The project uses FastAPI")

    provider = MockProvider({"facts": [
        {"content": "User prefers Python over JavaScript",
         "fact_type": "user_preference", "importance": 0.9, "confidence": 0.9},
        {"content": "Project uses FastAPI",
         "fact_type": "project_context", "importance": 0.7, "confidence": 0.9},
    ]})

    consolidator = Consolidator(
        session_id="test", episodic=episodic, semantic=semantic, provider=provider,
    )
    result = consolidator.consolidate()

    assert result.episodes_processed == 2
    assert result.facts_created == 2
    assert result.llm_calls_made == 1


def test_consolidate_marks_episodes_as_consolidated(stores):
    episodic, semantic = stores
    episodic.add("User message one")
    episodic.add("User message two")

    assert episodic.count_unconsolidated() == 2

    consolidator = Consolidator(
        session_id="test", episodic=episodic, semantic=semantic,
        provider=MockProvider({"facts": []}),
    )
    consolidator.consolidate()

    assert episodic.count_unconsolidated() == 0


def test_consolidate_uses_fallback_on_primary_failure(stores):
    episodic, semantic = stores
    episodic.add("User prefers dark mode")

    fallback = MockProvider({"facts": [
        {"content": "User prefers dark mode",
         "fact_type": "user_preference", "importance": 0.7, "confidence": 0.9},
    ]})

    consolidator = Consolidator(
        session_id="test", episodic=episodic, semantic=semantic,
        provider=FailingProvider(), fallback=fallback,
    )
    result = consolidator.consolidate()

    assert result.facts_created == 1
    assert len(result.errors) == 0


def test_consolidate_records_error_when_both_providers_fail(stores):
    episodic, semantic = stores
    episodic.add("Some message")

    consolidator = Consolidator(
        session_id="test", episodic=episodic, semantic=semantic,
        provider=FailingProvider(), fallback=None,
    )
    result = consolidator.consolidate()

    assert len(result.errors) > 0


# ── Mnemosyne.consolidate() end-to-end ───────────────────────────────────────

def test_mnemosyne_consolidate_facts_appear_in_recall(tmp_path):
    provider = MockProvider({"facts": [
        {"content": "User strongly prefers Python",
         "fact_type": "user_preference", "importance": 0.9, "confidence": 0.9},
    ]})

    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=provider,
    )

    memory.remember("I really prefer Python for all my projects")
    memory.consolidate()

    # Semantic fact should now be surfaced in recall
    raw = memory.recall("Python preferences", return_raw=True)
    from mnemosyne.types import SemanticFact
    semantic_results = [m for m in raw if isinstance(m, SemanticFact)]
    assert len(semantic_results) > 0
    assert any("Python" in m.content for m in semantic_results)
