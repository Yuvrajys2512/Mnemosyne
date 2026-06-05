"""Consolidation scenarios — does the LLM extraction produce good semantic facts?"""
from mnemosyne.eval.types import Scenario, Turn
from mnemosyne.types import FactType, SemanticFact


def _semantic_fact_count(min_facts: int):
    """Pass if at least min_facts SemanticFact objects were created."""
    def evaluator(memory, recall_output: str) -> float:
        facts = memory.semantic._collection.get(
            where={"session_id": f"eval_"}  # prefix won't match; use count()
        )
        count = memory.semantic._collection.count()
        if count >= min_facts:
            return 1.0
        return count / min_facts
    return evaluator


def _semantic_fact_keyword(keywords: list[str]):
    """Score = fraction of keywords present across all stored semantic facts."""
    def evaluator(memory, recall_output: str) -> float:
        results = memory.semantic._collection.get(include=["documents"])
        if not results["documents"]:
            return 0.0
        combined = " ".join(results["documents"]).lower()
        hits = sum(1 for kw in keywords if kw.lower() in combined)
        return hits / len(keywords) if keywords else 0.0
    return evaluator


SCENARIOS: list[Scenario] = [
    Scenario(
        name="consolidation_extracts_preference",
        description="Episodic preference event → semantic USER_PREFERENCE fact.",
        dimension="consolidation",
        turns=[
            Turn(role="user", content="I strongly prefer using type hints in Python; "
                 "I always annotate function signatures."),
            Turn(role="agent", content="Good practice.", timestamp_offset=2),
        ],
        evaluator=_semantic_fact_keyword(["python", "type"]),
        tags=["preference", "extraction"],
    ),
    Scenario(
        name="consolidation_extracts_project_context",
        description="Project context should be stored as a semantic fact.",
        dimension="consolidation",
        turns=[
            Turn(role="user", content="The project is a REST API built with FastAPI, "
                 "deployed on Kubernetes, and uses PostgreSQL 15."),
            Turn(role="agent", content="Got it.", timestamp_offset=2),
        ],
        evaluator=_semantic_fact_keyword(["fastapi", "postgresql"]),
        tags=["project", "extraction"],
    ),
    Scenario(
        name="consolidation_multi_turn_aggregation",
        description="Multiple turns should be consolidated into coherent facts.",
        dimension="consolidation",
        turns=[
            Turn(role="user", content="I always write tests before code — TDD is my workflow.",
                 timestamp_offset=0),
            Turn(role="agent", content="Understood.", timestamp_offset=5),
            Turn(role="user", content="I use pytest and always aim for 90%+ coverage.",
                 timestamp_offset=10),
        ],
        evaluator=_semantic_fact_keyword(["test", "pytest"]),
        tags=["multi-turn", "extraction"],
    ),
    Scenario(
        name="consolidation_creates_at_least_one_fact",
        description="Any meaningful conversation should produce ≥1 semantic fact.",
        dimension="consolidation",
        turns=[
            Turn(role="user", content="I'm working on a data pipeline that processes "
                 "1 million records per hour using Apache Kafka."),
            Turn(role="agent", content="Noted.", timestamp_offset=2),
        ],
        evaluator=_semantic_fact_count(1),
        tags=["extraction"],
    ),
]
