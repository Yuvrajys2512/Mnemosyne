"""Types for the Mnemosyne evaluation framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Turn:
    """One exchange in a simulated conversation."""
    role: str        # "user" or "agent"
    content: str
    timestamp_offset: float = 0.0   # seconds from scenario start


@dataclass
class Scenario:
    """A self-contained evaluation scenario."""
    name: str
    description: str
    dimension: str                  # recall_accuracy | consolidation | retention | edge_cases
    turns: list[Turn]
    # Called with (Mnemosyne instance, recall_output: str) → score in [0, 1]
    evaluator: Callable
    tags: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Outcome of running one scenario."""
    scenario_name: str
    dimension: str
    score: float                    # [0, 1]
    passed: bool                    # score >= pass_threshold (default 0.6)
    details: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated results across all scenarios."""
    results: list[ScenarioResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.total if self.total else 0.0

    def by_dimension(self) -> dict[str, list[ScenarioResult]]:
        dims: dict[str, list[ScenarioResult]] = {}
        for r in self.results:
            dims.setdefault(r.dimension, []).append(r)
        return dims
