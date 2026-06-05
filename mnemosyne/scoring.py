from __future__ import annotations

from dataclasses import dataclass

from mnemosyne.types import Memory


@dataclass
class ScoringWeights:
    semantic: float = 0.40
    recency: float = 0.30
    importance: float = 0.20
    frequency: float = 0.10

    def __post_init__(self) -> None:
        total = self.semantic + self.recency + self.importance + self.frequency
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


class ScoringEngine:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(self, memory: Memory, query_embedding: list[float]) -> float:
        raise NotImplementedError

    def rank(
        self,
        candidates: list[Memory],
        query_embedding: list[float],
    ) -> list[tuple[Memory, float]]:
        raise NotImplementedError
