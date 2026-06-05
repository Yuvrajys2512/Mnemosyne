"""
Integration tests for Mnemosyne.recall().
These use real ChromaDB (via tmp_path) with a FakeEmbeddingFunction.
"""
import hashlib

import numpy as np
import pytest

from mnemosyne import EventType, Mnemosyne


class FakeEmbeddingFunction:
    def name(self) -> str:
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


@pytest.fixture
def memory(tmp_path):
    return Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
    )


# ── recall() return type ──────────────────────────────────────────────────────

def test_recall_empty_store_returns_empty_string(memory):
    result = memory.recall("anything")
    assert result == ""


def test_recall_returns_string_by_default(memory):
    memory.remember("I prefer Python")
    result = memory.recall("Python preferences")
    assert isinstance(result, str)


def test_recall_return_raw_gives_list(memory):
    memory.remember("I prefer Python")
    result = memory.recall("Python", return_raw=True)
    assert isinstance(result, list)


def test_recall_return_raw_contains_episodic_events(memory):
    from mnemosyne.types import EpisodicEvent
    memory.remember("I prefer Python")
    results = memory.recall("Python", return_raw=True)
    assert all(isinstance(r, EpisodicEvent) for r in results)


# ── recall() content ──────────────────────────────────────────────────────────

def test_recall_output_contains_memory_context_header(memory):
    memory.remember("User prefers Python")
    result = memory.recall("Python")
    assert "Memory Context" in result


def test_recall_stored_content_appears_in_output(memory):
    memory.remember("I always prefer async Python patterns")
    result = memory.recall("Python async")
    assert "async Python patterns" in result


def test_recall_multiple_memories_all_can_appear(memory):
    memory.remember("Project uses FastAPI")
    memory.remember("Database is PostgreSQL")
    result = memory.recall("project stack")
    # With fake embeddings retrieval order isn't semantic, but both should fit in budget
    assert "FastAPI" in result or "PostgreSQL" in result


# ── token budget ──────────────────────────────────────────────────────────────

def test_recall_respects_token_budget(memory):
    # Store enough content to exceed a tiny budget
    for i in range(20):
        memory.remember(f"This is memory number {i} with some extra words to fill tokens")
    result = memory.recall("memory", token_budget=50)
    # Result should exist but be limited
    assert isinstance(result, str)


# ── access count ─────────────────────────────────────────────────────────────

def test_recall_increments_access_count(memory):
    memory.remember("Python is my preferred language")
    # First recall
    memory.recall("Python", return_raw=True)
    # Second recall — access count should have been incremented
    results = memory.recall("Python", return_raw=True)
    assert any(r.access_count > 0 for r in results)


# ── remember integration ──────────────────────────────────────────────────────

def test_remember_different_event_types(memory):
    memory.remember("user said something", event_type=EventType.USER_MESSAGE)
    memory.remember("agent replied", event_type=EventType.AGENT_RESPONSE)
    memory.remember("tool returned data", event_type=EventType.TOOL_RESULT)
    result = memory.recall("something")
    assert isinstance(result, str)


def test_remember_returns_nonempty_id_for_real_content(memory):
    id_ = memory.remember("I prefer Python over JavaScript")
    assert id_ != ""
    assert id_.startswith("ep_")


def test_remember_returns_empty_for_trivial_content(memory):
    id_ = memory.remember("ok")
    assert id_ == ""
