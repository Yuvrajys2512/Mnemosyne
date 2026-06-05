"""Retention scenarios — does important information survive across recall queries?"""
from mnemosyne.eval.types import Scenario, Turn

_ONE_HOUR = 3_600.0
_ONE_DAY = 86_400.0


def _contains_after_gap(keywords: list[str]):
    """Keywords must appear in recall even after a simulated time gap."""
    def evaluator(memory, recall_output: str) -> float:
        lower = recall_output.lower()
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        return hits / len(keywords) if keywords else 0.0
    return evaluator


SCENARIOS: list[Scenario] = [
    Scenario(
        name="retention_preference_survives_gap",
        description="User preference stated at t=0 should survive a 1-hour simulated gap.",
        dimension="retention",
        turns=[
            Turn(role="user", content="I always use black for Python code formatting.",
                 timestamp_offset=0),
            Turn(role="agent", content="Got it.", timestamp_offset=2),
            # Simulated 1-hour gap — later turn to fill context
            Turn(role="user", content="Help me set up pre-commit hooks.",
                 timestamp_offset=_ONE_HOUR),
        ],
        evaluator=_contains_after_gap(["black"]),
        tags=["retention", "preference"],
    ),
    Scenario(
        name="retention_project_context_survives_day",
        description="Project tech stack should still appear after a 24-hour gap.",
        dimension="retention",
        turns=[
            Turn(role="user", content="We're building a recommendation engine using PyTorch.",
                 timestamp_offset=0),
            Turn(role="agent", content="Understood.", timestamp_offset=2),
            Turn(role="user", content="Let's continue on the model architecture.",
                 timestamp_offset=_ONE_DAY),
        ],
        evaluator=_contains_after_gap(["pytorch"]),
        tags=["retention", "project"],
    ),
    Scenario(
        name="retention_background_persists",
        description="User background information should persist across the session.",
        dimension="retention",
        turns=[
            Turn(role="user", content="I have a PhD in computer science specialising in distributed systems.",
                 timestamp_offset=0),
            Turn(role="agent", content="Impressive background.", timestamp_offset=2),
            Turn(role="user", content="What do you think about Raft consensus?",
                 timestamp_offset=300),
        ],
        evaluator=_contains_after_gap(["phd", "distributed"]),
        tags=["retention", "background"],
    ),
    Scenario(
        name="retention_constraint_persists",
        description="Task constraint should remain visible throughout session.",
        dimension="retention",
        turns=[
            Turn(role="user", content="Never use async/await in this codebase — it's a strict requirement.",
                 timestamp_offset=0),
            Turn(role="agent", content="Noted, I'll keep the code synchronous.", timestamp_offset=2),
            Turn(role="user", content="Now write a database query function.", timestamp_offset=100),
        ],
        evaluator=_contains_after_gap(["async", "synchronous"]),
        tags=["retention", "constraint"],
    ),
]
