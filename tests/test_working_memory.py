"""
Tests for WorkingMemory — tested in isolation using hand-built EpisodicEvent
objects so there's no dependency on ChromaDB here.
"""
import time

import pytest

from mnemosyne.core.working import WorkingMemory, _estimate_tokens, _format_age
from mnemosyne.types import EpisodicEvent, EventType, FactType, SemanticFact


# ── helpers ───────────────────────────────────────────────────────────────────

def make_event(content: str, event_type=EventType.USER_MESSAGE, age_seconds=0.0) -> EpisodicEvent:
    return EpisodicEvent(
        content=content,
        session_id="test",
        event_type=event_type,
        timestamp=time.time() - age_seconds,
    )


def make_fact(content: str, fact_type=FactType.USER_PREFERENCE) -> SemanticFact:
    return SemanticFact(content=content, session_id="test", fact_type=fact_type)


@pytest.fixture
def wm():
    return WorkingMemory(token_budget=500)


# ── _estimate_tokens ──────────────────────────────────────────────────────────

def test_estimate_tokens_scales_with_length():
    short = _estimate_tokens("hello")
    long = _estimate_tokens("hello " * 100)
    assert long > short


def test_estimate_tokens_nonzero_for_nonempty():
    assert _estimate_tokens("any text") > 0


# ── _format_age ───────────────────────────────────────────────────────────────

def test_format_age_seconds():
    assert "s ago" in _format_age(time.time() - 30)


def test_format_age_minutes():
    assert "m ago" in _format_age(time.time() - 300)


def test_format_age_hours():
    assert "h ago" in _format_age(time.time() - 7200)


def test_format_age_days():
    assert "d ago" in _format_age(time.time() - 172800)


# ── select() ─────────────────────────────────────────────────────────────────

def test_select_empty_candidates_returns_empty(wm):
    assert wm.select([]) == []


def test_select_returns_candidates_that_fit(wm):
    events = [make_event("short") for _ in range(3)]
    selected = wm.select(events)
    assert len(selected) == 3


def test_select_respects_token_budget():
    wm_tiny = WorkingMemory(token_budget=10)
    # Each "word " * 50 is ~50 chars → ~15 tokens each, exceeds a 10-token budget
    events = [make_event("word " * 50) for _ in range(5)]
    selected = wm_tiny.select(events)
    assert len(selected) == 0


def test_select_skips_oversized_but_keeps_smaller():
    wm_small = WorkingMemory(token_budget=30)
    big = make_event("x " * 100)     # ~30 tokens, too big
    small = make_event("hello world") # ~4 tokens, fits
    selected = wm_small.select([big, small])
    assert small in selected
    assert big not in selected


def test_select_preserves_input_order(wm):
    events = [make_event(f"event {i}") for i in range(5)]
    selected = wm.select(events)
    # Order of input list should be preserved in output
    contents = [e.content for e in selected]
    assert contents == [e.content for e in events[:len(selected)]]


def test_select_with_explicit_budget_overrides_default():
    wm = WorkingMemory(token_budget=1000)
    events = [make_event("hello " * 20) for _ in range(10)]
    # Very tight explicit budget
    selected = wm.select(events, token_budget=5)
    assert len(selected) == 0


# ── format_for_prompt() ───────────────────────────────────────────────────────

def test_format_empty_returns_empty_string(wm):
    assert wm.format_for_prompt([]) == ""


def test_format_returns_string(wm):
    events = [make_event("User asked about Python")]
    result = wm.format_for_prompt(events)
    assert isinstance(result, str)


def test_format_contains_memory_context_header(wm):
    result = wm.format_for_prompt([make_event("something")])
    assert "Memory Context" in result


def test_format_episodic_events_appear_in_recent_section(wm):
    event = make_event("I prefer async patterns")
    result = wm.format_for_prompt([event])
    assert "Recent Context" in result
    assert "async patterns" in result


def test_format_user_message_has_user_prefix(wm):
    event = make_event("hello there", event_type=EventType.USER_MESSAGE)
    result = wm.format_for_prompt([event])
    assert "User:" in result


def test_format_agent_response_has_agent_prefix(wm):
    event = make_event("here is the answer", event_type=EventType.AGENT_RESPONSE)
    result = wm.format_for_prompt([event])
    assert "Agent:" in result


def test_format_semantic_preference_in_preferences_section(wm):
    fact = make_fact("User prefers Python", fact_type=FactType.USER_PREFERENCE)
    result = wm.format_for_prompt([fact])
    assert "User Preferences" in result
    assert "Python" in result


def test_format_semantic_project_context_in_project_section(wm):
    fact = make_fact("Project uses FastAPI", fact_type=FactType.PROJECT_CONTEXT)
    result = wm.format_for_prompt([fact])
    assert "Project Context" in result
    assert "FastAPI" in result


def test_format_recent_events_sorted_oldest_first(wm):
    old_event = make_event("old message", age_seconds=300)
    new_event = make_event("new message", age_seconds=10)
    result = wm.format_for_prompt([new_event, old_event])  # reversed input
    old_pos = result.index("old message")
    new_pos = result.index("new message")
    assert old_pos < new_pos  # old should appear first in the block


def test_format_long_content_is_truncated(wm):
    event = make_event("x" * 500)
    result = wm.format_for_prompt([event])
    # Should have truncated with ellipsis
    assert "…" in result


def test_format_short_content_is_not_truncated(wm):
    event = make_event("short content")
    result = wm.format_for_prompt([event])
    assert "…" not in result
