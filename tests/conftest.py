"""Shared pytest fixtures and test helpers."""
import hashlib

import numpy as np
import pytest


class FakeEmbeddingFunction:
    """Deterministic, hash-based fake embedding function.

    Produces a stable 384-dim vector for any text string using MD5 seeding,
    so identical text always yields the same vector — no model download needed.
    The `name()` method returns "default" so ChromaDB skips the conflict check.
    """

    def name(self):
        return "default"

    def _embed(self, texts):
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            results.append(rng.standard_normal(384).tolist())
        return results

    def __call__(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input)


@pytest.fixture
def fake_ef():
    return FakeEmbeddingFunction()
