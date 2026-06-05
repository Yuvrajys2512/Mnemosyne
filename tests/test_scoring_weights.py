import pytest
from mnemosyne.scoring import ScoringWeights


def test_default_weights_sum_to_one():
    w = ScoringWeights()
    assert abs(w.semantic + w.recency + w.importance + w.frequency - 1.0) < 1e-6


def test_custom_weights_sum_to_one():
    w = ScoringWeights(semantic=0.45, recency=0.25, importance=0.25, frequency=0.05)
    assert abs(w.semantic + w.recency + w.importance + w.frequency - 1.0) < 1e-6


def test_invalid_weights_raise():
    with pytest.raises(AssertionError):
        ScoringWeights(semantic=0.5, recency=0.5, importance=0.5, frequency=0.5)
