"""Tests for async consolidation and auto-trigger behaviour."""
import asyncio
import hashlib
import json

import numpy as np
import pytest

from mnemosyne import Mnemosyne
from mnemosyne.providers import LLMProvider


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
    def __init__(self, facts=None):
        self._facts = facts or []

    @property
    def name(self):
        return "mock"

    def complete(self, system, user):
        return json.dumps({"facts": self._facts})


# ── aconsolidate() ────────────────────────────────────────────────────────────

async def test_aconsolidate_returns_result(tmp_path):
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
    )
    memory.remember("I prefer Python")
    result = await memory.aconsolidate()
    assert result.episodes_processed == 1


async def test_aconsolidate_marks_episodes_consolidated(tmp_path):
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
    )
    memory.remember("message one")
    memory.remember("message two")
    assert memory.episodic.count_unconsolidated() == 2

    await memory.aconsolidate()
    assert memory.episodic.count_unconsolidated() == 0


async def test_aconsolidate_is_idempotent(tmp_path):
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
    )
    memory.remember("hello world")
    await memory.aconsolidate()
    result2 = await memory.aconsolidate()
    assert result2.episodes_processed == 0


# ── auto-trigger ──────────────────────────────────────────────────────────────

async def test_auto_consolidation_triggers_after_threshold(tmp_path):
    """When enough events accumulate, background consolidation fires automatically."""
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
        auto_consolidate_threshold=3,
    )

    # Add events inside an async context so get_running_loop() works
    for i in range(4):
        memory.remember(f"this is event number {i} with real content")

    # Give the background task a moment to complete
    await asyncio.sleep(0.2)

    assert memory.episodic.count_unconsolidated() == 0


async def test_auto_consolidation_does_not_trigger_below_threshold(tmp_path):
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
        auto_consolidate_threshold=10,
    )
    for i in range(5):
        memory.remember(f"event {i} content here")

    await asyncio.sleep(0.05)
    # Threshold not hit — should still be unconsolidated
    assert memory.episodic.count_unconsolidated() == 5


# ── sync context graceful skip ────────────────────────────────────────────────

def test_remember_in_sync_context_does_not_raise(tmp_path):
    """In a sync context, auto-trigger is silently skipped — no crash."""
    memory = Mnemosyne(
        session_id="test",
        storage_path=str(tmp_path),
        _embedding_function=FakeEmbeddingFunction(),
        _provider=MockProvider(),
        auto_consolidate_threshold=1,
    )
    # This should not raise even though threshold is immediately exceeded
    memory.remember("first message with real content")
    memory.remember("second message with real content")
