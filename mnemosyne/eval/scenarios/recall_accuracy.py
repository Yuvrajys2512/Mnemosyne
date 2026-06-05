"""Recall accuracy scenarios — does the right information surface?"""
from mnemosyne.eval.types import Scenario, Turn


def _contains(keywords: list[str]):
    """Score = fraction of keywords present in recall output (case-insensitive)."""
    def evaluator(memory, recall_output: str) -> float:
        lower = recall_output.lower()
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        return hits / len(keywords) if keywords else 0.0
    return evaluator


SCENARIOS: list[Scenario] = [
    Scenario(
        name="preference_single_fact",
        description="User states one preference; it should appear in recall.",
        dimension="recall_accuracy",
        turns=[
            Turn(role="user", content="I always prefer Python over JavaScript."),
            Turn(role="agent", content="Noted, I'll keep that in mind.", timestamp_offset=2),
        ],
        evaluator=_contains(["python"]),
        tags=["preference"],
    ),
    Scenario(
        name="preference_multiple_facts",
        description="User states two preferences; both should surface.",
        dimension="recall_accuracy",
        turns=[
            Turn(role="user", content="I prefer Python and I always use FastAPI for backend work."),
            Turn(role="agent", content="Got it.", timestamp_offset=2),
        ],
        evaluator=_contains(["python", "fastapi"]),
        tags=["preference"],
    ),
    Scenario(
        name="project_context",
        description="Project tech stack mentioned; should appear in recall.",
        dimension="recall_accuracy",
        turns=[
            Turn(role="user", content="This project uses PostgreSQL and Redis for caching."),
            Turn(role="agent", content="Understood.", timestamp_offset=2),
            Turn(role="user", content="We're deploying to AWS Lambda.", timestamp_offset=10),
        ],
        evaluator=_contains(["postgresql", "redis"]),
        tags=["project"],
    ),
    Scenario(
        name="user_background",
        description="User shares professional background; should be recalled.",
        dimension="recall_accuracy",
        turns=[
            Turn(role="user", content="I'm a senior machine learning engineer with 8 years of experience."),
            Turn(role="agent", content="Great, I'll tailor my explanations accordingly.", timestamp_offset=2),
        ],
        evaluator=_contains(["machine learning", "engineer"]),
        tags=["background"],
    ),
    Scenario(
        name="correction_overrides_original",
        description="User corrects earlier statement; correction should dominate recall.",
        dimension="recall_accuracy",
        turns=[
            Turn(role="user", content="We're using MySQL for the database.", timestamp_offset=0),
            Turn(role="agent", content="OK.", timestamp_offset=2),
            Turn(role="user", content="Actually, we migrated to PostgreSQL, not MySQL.", timestamp_offset=10),
        ],
        evaluator=_contains(["postgresql"]),
        tags=["correction"],
    ),
]
