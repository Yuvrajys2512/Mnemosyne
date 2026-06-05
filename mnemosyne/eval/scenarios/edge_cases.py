"""Edge case scenarios — graceful handling of unusual or adversarial inputs."""
from mnemosyne.eval.types import Scenario, Turn


def _no_crash(expected_keywords: list[str] | None = None):
    """Pass (score=1.0) if no exception was raised. Optionally check keywords."""
    def evaluator(memory, recall_output: str) -> float:
        if not isinstance(recall_output, str):
            return 0.0
        if expected_keywords:
            lower = recall_output.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
            keyword_score = hits / len(expected_keywords)
            return 0.5 + 0.5 * keyword_score
        return 1.0
    return evaluator


def _recall_is_non_empty():
    """Score = 1 if recall returns non-empty string, 0 otherwise."""
    def evaluator(memory, recall_output: str) -> float:
        return 1.0 if recall_output and recall_output.strip() else 0.0
    return evaluator


def _recall_is_empty_or_minimal():
    """Score = 1 if recall returns nothing meaningful (trivial input case)."""
    def evaluator(memory, recall_output: str) -> float:
        # An empty or very short output means trivial inputs were correctly dropped
        return 1.0 if len(recall_output.strip()) < 50 else 0.5
    return evaluator


SCENARIOS: list[Scenario] = [
    Scenario(
        name="edge_empty_session_recall",
        description="Recall on an empty session should return empty string, not crash.",
        dimension="edge_cases",
        turns=[],   # no turns — empty session
        evaluator=lambda memory, recall: 1.0 if isinstance(recall, str) else 0.0,
        tags=["empty", "robustness"],
    ),
    Scenario(
        name="edge_trivial_inputs_filtered",
        description="Trivial messages ('ok', 'yes') should not pollute memory.",
        dimension="edge_cases",
        turns=[
            Turn(role="user", content="ok"),
            Turn(role="user", content="yes"),
            Turn(role="user", content="sure"),
            Turn(role="agent", content="ok"),
        ],
        evaluator=_recall_is_empty_or_minimal(),
        tags=["filtering", "trivial"],
    ),
    Scenario(
        name="edge_very_long_input",
        description="Content >2000 chars should be stored (truncated) without crashing.",
        dimension="edge_cases",
        turns=[
            Turn(role="user", content="I need you to remember this: " + ("detailed requirement " * 200)),
        ],
        evaluator=_no_crash(),
        tags=["truncation", "robustness"],
    ),
    Scenario(
        name="edge_unicode_content",
        description="Unicode content (emoji, CJK characters) should not crash the pipeline.",
        dimension="edge_cases",
        turns=[
            Turn(role="user", content="My name is 田中 and I prefer writing code with 🐍 Python."),
            Turn(role="agent", content="こんにちは、田中さん。", timestamp_offset=2),
        ],
        evaluator=_no_crash(["python"]),
        tags=["unicode", "robustness"],
    ),
    Scenario(
        name="edge_duplicate_identical_messages",
        description="Identical messages added twice should not corrupt the memory store.",
        dimension="edge_cases",
        turns=[
            Turn(role="user", content="I prefer vim over emacs.", timestamp_offset=0),
            Turn(role="user", content="I prefer vim over emacs.", timestamp_offset=1),
            Turn(role="user", content="I prefer vim over emacs.", timestamp_offset=2),
        ],
        evaluator=_no_crash(["vim"]),
        tags=["dedup", "robustness"],
    ),
    Scenario(
        name="edge_numeric_only_content",
        description="Numbers-only messages should be stored and recallable.",
        dimension="edge_cases",
        turns=[
            Turn(role="user", content="The port number to use is 8443 and the timeout is 30 seconds."),
        ],
        evaluator=_no_crash(["8443"]),
        tags=["numeric", "robustness"],
    ),
]
