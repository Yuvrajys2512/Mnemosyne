"""Eval runner — executes scenarios against a Mnemosyne instance."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from mnemosyne import Mnemosyne
from mnemosyne.types import EventType

from .types import EvalReport, Scenario, ScenarioResult

if TYPE_CHECKING:
    from mnemosyne.providers import LLMProvider

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.6


class FakeEmbeddingFunction:
    """Deterministic embeddings for eval runs — no model download needed."""

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


def run_scenario(
    scenario: Scenario,
    provider: "LLMProvider",
    storage_path: str,
    pass_threshold: float = PASS_THRESHOLD,
    consolidate_after_turns: bool = True,
) -> ScenarioResult:
    """Run a single scenario and return its result."""
    t0 = time.monotonic()
    try:
        memory = Mnemosyne(
            session_id=f"eval_{scenario.name}",
            storage_path=storage_path,
            _embedding_function=FakeEmbeddingFunction(),
            _provider=provider,
        )

        base_time = time.time()
        for turn in scenario.turns:
            ts = base_time + turn.timestamp_offset
            memory.episodic.add(
                content=turn.content,
                event_type=(
                    EventType.USER_MESSAGE
                    if turn.role == "user"
                    else EventType.AGENT_RESPONSE
                ),
                _timestamp=ts,
            )

        if consolidate_after_turns:
            memory.consolidate()

        recall_output = memory.recall(
            query=" ".join(t.content for t in scenario.turns[:3]),
            token_budget=2000,
        )

        score = float(scenario.evaluator(memory, recall_output))
        score = max(0.0, min(1.0, score))

        return ScenarioResult(
            scenario_name=scenario.name,
            dimension=scenario.dimension,
            score=score,
            passed=score >= pass_threshold,
            details={"duration_s": round(time.monotonic() - t0, 3)},
        )

    except Exception as exc:
        logger.warning("Scenario %s failed: %s", scenario.name, exc)
        return ScenarioResult(
            scenario_name=scenario.name,
            dimension=scenario.dimension,
            score=0.0,
            passed=False,
            error=str(exc),
            details={"duration_s": round(time.monotonic() - t0, 3)},
        )


def run_eval(
    scenarios: list[Scenario],
    provider: "LLMProvider",
    storage_path: str,
    pass_threshold: float = PASS_THRESHOLD,
    consolidate_after_turns: bool = True,
) -> EvalReport:
    """Run all scenarios and return an aggregated report."""
    results = []
    for i, scenario in enumerate(scenarios, 1):
        logger.info("[%d/%d] %s", i, len(scenarios), scenario.name)
        result = run_scenario(
            scenario,
            provider=provider,
            storage_path=storage_path,
            pass_threshold=pass_threshold,
            consolidate_after_turns=consolidate_after_turns,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info("  → %s  score=%.2f", status, result.score)

    return EvalReport(results=results)
