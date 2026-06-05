"""Tests for SemanticStore — uses FakeEmbeddingFunction for speed."""
import hashlib
import time

import numpy as np
import pytest

from mnemosyne.core.semantic import SemanticStore
from mnemosyne.types import FactType, SemanticFact


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


def make_fact(
    content: str = "User prefers Python",
    fact_type: FactType = FactType.USER_PREFERENCE,
    importance: float = 0.7,
    session_id: str = "test",
) -> SemanticFact:
    return SemanticFact(
        content=content,
        session_id=session_id,
        fact_type=fact_type,
        importance=importance,
    )


@pytest.fixture
def store(tmp_path):
    return SemanticStore(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
    )


# ── add() ─────────────────────────────────────────────────────────────────────

def test_add_returns_fact_id(store):
    fact = make_fact()
    id_ = store.add(fact)
    assert id_.startswith("sem_")


def test_add_different_facts_have_unique_ids(store):
    a = store.add(make_fact("fact a"))
    b = store.add(make_fact("fact b"))
    assert a != b


# ── query() ───────────────────────────────────────────────────────────────────

def test_query_empty_store_returns_empty(store):
    assert store.query("anything") == []


def test_query_returns_tuples_with_similarity(store):
    store.add(make_fact("User prefers Python"))
    results = store.query("Python preferences")
    assert len(results) > 0
    fact, sim = results[0]
    assert hasattr(fact, "content")
    assert 0.0 <= sim <= 1.0


def test_query_excludes_superseded_facts(store):
    old = make_fact("User prefers JavaScript")
    new = make_fact("User prefers TypeScript")
    old_id = store.add(old)
    new_id = store.add(new)
    store.update(old_id, contradicted_by=new_id)

    results = store.query("preferred language")
    ids = [f.id for f, _ in results]
    assert old_id not in ids


# ── update() ─────────────────────────────────────────────────────────────────

def test_update_importance(store):
    fact = make_fact(importance=0.5)
    id_ = store.add(fact)
    store.update(id_, importance=0.9)

    results = store.query("Python")
    updated = next(f for f, _ in results if f.id == id_)
    assert updated.importance == pytest.approx(0.9)


def test_update_content_changes_document(store):
    fact = make_fact("User prefers Python")
    id_ = store.add(fact)
    store.update(id_, content="User prefers Python 3.11 specifically")

    results = store.query("Python version")
    updated = next((f for f, _ in results if f.id == id_), None)
    assert updated is not None
    assert "3.11" in updated.content


def test_update_source_episode_ids(store):
    fact = make_fact()
    id_ = store.add(fact)
    store.update(id_, source_episode_ids=["ep_abc", "ep_def"])

    results = store.query("Python")
    updated = next(f for f, _ in results if f.id == id_)
    assert "ep_abc" in updated.source_episode_ids


# ── supersede() ───────────────────────────────────────────────────────────────

def test_supersede_marks_old_as_contradicted(store):
    old = make_fact("User prefers Python 2")
    new_fact = make_fact("User prefers Python 3")
    old_id = store.add(old)
    store.supersede(old_id, new_fact)

    results = store.query("Python version")
    ids = [f.id for f, _ in results]
    assert old_id not in ids


def test_supersede_new_fact_is_retrievable(store):
    old = make_fact("User prefers Python 2")
    new_fact = make_fact("User prefers Python 3")
    old_id = store.add(old)
    new_id = store.supersede(old_id, new_fact)

    results = store.query("Python version")
    ids = [f.id for f, _ in results]
    assert new_id in ids


# ── upsert() ─────────────────────────────────────────────────────────────────

def test_upsert_new_fact_creates_entry(store):
    fact = make_fact("User prefers async Python")
    store.upsert(fact)
    results = store.query("async Python")
    assert len(results) > 0


def test_upsert_is_idempotent_for_new_facts(store):
    # Upserting a brand-new fact twice should result in one entry
    # (the second is similar enough to reinforce the first)
    fact1 = make_fact("User dislikes JavaScript frameworks")
    fact2 = make_fact("User dislikes JavaScript frameworks")
    store.upsert(fact1)
    store.upsert(fact2)
    # Both have identical content → second reinforces first, not a new entry
    assert store._collection.count() == 1


# ── increment_access() ────────────────────────────────────────────────────────

def test_increment_access_updates_count(store):
    fact = make_fact()
    id_ = store.add(fact)
    store.increment_access([id_], [3])

    results = store.query("Python")
    updated = next(f for f, _ in results if f.id == id_)
    assert updated.access_count == 3


# ── timestamp property ────────────────────────────────────────────────────────

def test_semantic_fact_timestamp_equals_last_updated():
    fact = make_fact()
    assert fact.timestamp == fact.last_updated
