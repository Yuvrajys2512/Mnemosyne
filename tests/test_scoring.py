"""
Tests for ScoringEngine and its component functions.

Key test: the A > C > B example from docs/07-scoring-function.md —
verifying that composite scoring beats raw cosine similarity.
"""
import math
import time

import pytest

from mnemosyne.scoring import (
    ScoringEngine,
    ScoringWeights,
    frequency_score,
    half_life_for,
    importance_score,
    recency_score,
)
from mnemosyne.types import EpisodicEvent, EventType, FactType, SemanticFact


# ── helpers ───────────────────────────────────────────────────────────────────

def make_event(
    content: str = "test",
    event_type: EventType = EventType.USER_MESSAGE,
    age_seconds: float = 0.0,
    importance: float = 0.5,
    access_count: int = 0,
) -> EpisodicEvent:
    return EpisodicEvent(
        content=content,
        session_id="test",
        event_type=event_type,
        timestamp=time.time() - age_seconds,
        importance=importance,
        access_count=access_count,
    )


def make_fact(
    content: str = "test",
    fact_type: FactType = FactType.USER_PREFERENCE,
    age_seconds: float = 0.0,
    importance: float = 0.5,
    access_count: int = 0,
) -> SemanticFact:
    return SemanticFact(
        content=content,
        session_id="test",
        fact_type=fact_type,
        created_at=time.time() - age_seconds,
        last_updated=time.time() - age_seconds,
        importance=importance,
        access_count=access_count,
    )


# ── recency_score ─────────────────────────────────────────────────────────────

def test_recency_brand_new_scores_one():
    score = recency_score(time.time(), time.time(), half_life=3600)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_recency_at_half_life_scores_half():
    t = time.time()
    score = recency_score(t - 3600, t, half_life=3600)
    assert score == pytest.approx(0.5, abs=1e-3)


def test_recency_at_two_half_lives_scores_quarter():
    t = time.time()
    score = recency_score(t - 7200, t, half_life=3600)
    assert score == pytest.approx(0.25, abs=1e-3)


def test_recency_older_memory_scores_lower():
    t = time.time()
    recent = recency_score(t - 60, t, half_life=3600)
    old = recency_score(t - 7200, t, half_life=3600)
    assert recent > old


def test_recency_stays_in_0_1():
    t = time.time()
    for age in [0, 60, 3600, 86400, 2592000]:
        s = recency_score(t - age, t, half_life=3600)
        assert 0.0 <= s <= 1.0


# ── importance_score ──────────────────────────────────────────────────────────

def test_importance_score_returns_stored_value():
    m = make_event(importance=0.8)
    assert importance_score(m) == pytest.approx(0.8)


def test_importance_score_low():
    m = make_event(importance=0.1)
    assert importance_score(m) == pytest.approx(0.1)


# ── frequency_score ───────────────────────────────────────────────────────────

def test_frequency_zero_access_is_zero():
    assert frequency_score(0) == 0.0


def test_frequency_one_access_is_nonzero():
    assert frequency_score(1) > 0.0


def test_frequency_higher_access_scores_higher():
    assert frequency_score(10) > frequency_score(1)


def test_frequency_caps_at_one():
    assert frequency_score(1_000_000) == pytest.approx(1.0)


def test_frequency_log_diminishing_returns():
    # Difference between 1 and 2 should be larger than between 50 and 51
    delta_low = frequency_score(2) - frequency_score(1)
    delta_high = frequency_score(51) - frequency_score(50)
    assert delta_low > delta_high


# ── half_life_for ─────────────────────────────────────────────────────────────

def test_half_life_user_message_is_one_hour():
    m = make_event(event_type=EventType.USER_MESSAGE)
    assert half_life_for(m) == 3_600


def test_half_life_user_preference_is_thirty_days():
    m = make_fact(fact_type=FactType.USER_PREFERENCE)
    assert half_life_for(m) == 2_592_000


def test_half_life_semantic_always_longer_than_episodic():
    episodic = make_event(event_type=EventType.USER_MESSAGE)
    semantic = make_fact(fact_type=FactType.USER_PREFERENCE)
    assert half_life_for(semantic) > half_life_for(episodic)


# ── ScoringEngine.score() ─────────────────────────────────────────────────────

def test_score_is_in_0_1():
    engine = ScoringEngine()
    m = make_event(importance=0.5, access_count=0, age_seconds=60)
    s = engine.score(m, semantic_sim=0.7)
    assert 0.0 <= s <= 1.0


def test_score_higher_similarity_scores_higher():
    engine = ScoringEngine()
    t = time.time()
    m = make_event(age_seconds=0)
    low = engine.score(m, semantic_sim=0.3, t_now=t)
    high = engine.score(m, semantic_sim=0.9, t_now=t)
    assert high > low


def test_score_higher_importance_scores_higher():
    engine = ScoringEngine()
    t = time.time()
    low_imp = make_event(importance=0.2, age_seconds=0)
    high_imp = make_event(importance=0.9, age_seconds=0)
    assert engine.score(high_imp, 0.7, t) > engine.score(low_imp, 0.7, t)


def test_score_more_recent_scores_higher():
    engine = ScoringEngine()
    t = time.time()
    old = make_event(age_seconds=86400, importance=0.5)
    new = make_event(age_seconds=10, importance=0.5)
    assert engine.score(new, 0.7, t) > engine.score(old, 0.7, t)


# ── The A > C > B motivating example ─────────────────────────────────────────
#
# This is the core test from docs/07-scoring-function.md.
# Pure cosine similarity ranks B > A > C.
# Composite scoring should rank A > C > B.
#
# Memory A: user preference fact, 10 min old, sim=0.72, importance=0.8
# Memory B: user preference fact, 3 months old, sim=0.81, importance=0.75
# Memory C: episodic event, 30 sec old, sim=0.70, importance=0.5

def test_composite_score_beats_raw_similarity():
    engine = ScoringEngine()
    t = time.time()

    # Semantic facts for A and B (long half-life = recency stays near 1.0 for recent ones)
    memory_a = make_fact(
        content="User prefers async Python patterns",
        fact_type=FactType.USER_PREFERENCE,
        age_seconds=600,       # 10 minutes old
        importance=0.8,
    )
    memory_b = make_fact(
        content="User hates JavaScript",
        fact_type=FactType.USER_PREFERENCE,
        age_seconds=90 * 86_400,  # 3 months old
        importance=0.75,
    )
    # Episodic event for C (short half-life but brand new)
    memory_c = make_event(
        content="User is debugging async code",
        event_type=EventType.USER_MESSAGE,
        age_seconds=30,        # 30 seconds old
        importance=0.5,
    )

    sim_a, sim_b, sim_c = 0.72, 0.81, 0.70

    score_a = engine.score(memory_a, sim_a, t_now=t)
    score_b = engine.score(memory_b, sim_b, t_now=t)
    score_c = engine.score(memory_c, sim_c, t_now=t)

    # Pure cosine sim would rank B > A > C
    # Composite scoring should rank A > C > B
    assert score_a > score_c, f"Expected A({score_a:.3f}) > C({score_c:.3f})"
    assert score_c > score_b, f"Expected C({score_c:.3f}) > B({score_b:.3f})"


# ── ScoringEngine.rank() ──────────────────────────────────────────────────────

def test_rank_returns_sorted_descending():
    engine = ScoringEngine()
    t = time.time()
    candidates = [
        (make_event(age_seconds=7200, importance=0.3), 0.5),
        (make_event(age_seconds=10, importance=0.9), 0.9),
        (make_event(age_seconds=60, importance=0.6), 0.7),
    ]
    ranked = engine.rank(candidates, t_now=t)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_returns_empty():
    engine = ScoringEngine()
    assert engine.rank([]) == []


def test_rank_returns_same_count_as_input():
    engine = ScoringEngine()
    candidates = [(make_event(), 0.5) for _ in range(5)]
    ranked = engine.rank(candidates)
    assert len(ranked) == 5


def test_rank_stale_high_similarity_loses_to_fresh_lower_similarity():
    """A very old memory with high similarity should lose to a recent one."""
    engine = ScoringEngine()
    t = time.time()

    stale = make_fact(
        fact_type=FactType.USER_PREFERENCE,
        age_seconds=180 * 86_400,  # 6 months old
        importance=0.5,
    )
    fresh = make_event(
        age_seconds=30,
        importance=0.5,
    )

    candidates = [(stale, 0.95), (fresh, 0.60)]
    ranked = engine.rank(candidates, t_now=t)

    top_memory = ranked[0][0]
    assert top_memory is fresh
