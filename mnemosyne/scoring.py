from __future__ import annotations

import math
import time
from dataclasses import dataclass

from mnemosyne.types import EpisodicEvent, EventType, FactType, Memory, SemanticFact

# How long until a memory's recency score halves.
# Episodic events decay fast (they're meant for short-term relevance).
# Semantic facts decay slowly (they're durable knowledge).
_HALF_LIVES: dict[EventType | FactType, float] = {
    # Episodic
    EventType.USER_MESSAGE:    3_600,       # 1 hour
    EventType.AGENT_RESPONSE:  1_800,       # 30 minutes
    EventType.TOOL_CALL:         900,       # 15 minutes
    EventType.TOOL_RESULT:       900,       # 15 minutes
    EventType.OBSERVATION:     3_600,       # 1 hour
    EventType.SYSTEM:          1_800,       # 30 minutes
    # Semantic
    FactType.USER_PREFERENCE:  2_592_000,   # 30 days
    FactType.USER_BACKGROUND:  2_592_000,   # 30 days
    FactType.PROJECT_CONTEXT:    604_800,   # 7 days
    FactType.TASK_CONSTRAINT:  2_592_000,   # 30 days
    FactType.ESTABLISHED_FACT:   604_800,   # 7 days
    FactType.DECISION:         1_209_600,   # 14 days
    FactType.OPEN_QUESTION:      259_200,   # 3 days
}
_DEFAULT_HALF_LIFE = 86_400  # 1 day fallback


@dataclass
class ScoringWeights:
    semantic: float = 0.40
    recency: float = 0.30
    importance: float = 0.20
    frequency: float = 0.10

    def __post_init__(self) -> None:
        total = self.semantic + self.recency + self.importance + self.frequency
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


# ── Component functions (pure, testable in isolation) ─────────────────────────

def recency_score(timestamp: float, t_now: float, half_life: float) -> float:
    """Exponential decay: 1.0 when brand new, halves every `half_life` seconds."""
    age = max(0.0, t_now - timestamp)
    return 2.0 ** (-age / half_life)


def importance_score(memory: Memory) -> float:
    """The stored importance scalar, already in [0, 1]."""
    return float(memory.importance)


def frequency_score(access_count: int) -> float:
    """Log-normalized: heavy diminishing returns past the first few retrievals."""
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log(access_count + 1) / math.log(100))


def half_life_for(memory: Memory) -> float:
    """Look up the decay rate for this memory type."""
    if isinstance(memory, EpisodicEvent):
        return _HALF_LIVES.get(memory.event_type, _DEFAULT_HALF_LIFE)
    if isinstance(memory, SemanticFact):
        return _HALF_LIVES.get(memory.fact_type, _DEFAULT_HALF_LIFE)
    return _DEFAULT_HALF_LIFE


# ── ScoringEngine ─────────────────────────────────────────────────────────────

class ScoringEngine:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(
        self,
        memory: Memory,
        semantic_sim: float,
        t_now: float | None = None,
    ) -> float:
        """
        Composite score in [0, 1].

        semantic_sim is the cosine similarity already in [0, 1] (converted from
        ChromaDB's distance: sim = 1 - distance).
        """
        if t_now is None:
            t_now = time.time()

        w = self.weights
        s_sim = float(semantic_sim)
        s_rec = recency_score(memory.timestamp, t_now, half_life_for(memory))
        s_imp = importance_score(memory)
        s_frq = frequency_score(memory.access_count)

        return (
            w.semantic   * s_sim
            + w.recency  * s_rec
            + w.importance * s_imp
            + w.frequency  * s_frq
        )

    def rank(
        self,
        candidates: list[tuple[Memory, float]],
        t_now: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Re-rank (memory, semantic_sim) pairs by composite score.
        Returns list of (memory, composite_score) sorted descending.
        """
        if t_now is None:
            t_now = time.time()

        scored = [
            (memory, self.score(memory, sim, t_now))
            for memory, sim in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
