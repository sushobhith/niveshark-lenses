"""Tests for the Health Score composite calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.health_score import (
    HealthScoreResult,
    compute_health_score,
    _score_to_grade,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Unit tests: grade mapping
# ---------------------------------------------------------------------------

def test_grade_a():
    assert _score_to_grade(80) == "A"
    assert _score_to_grade(95) == "A"
    assert _score_to_grade(100) == "A"


def test_grade_b():
    assert _score_to_grade(65) == "B"
    assert _score_to_grade(79.9) == "B"


def test_grade_c():
    assert _score_to_grade(50) == "C"
    assert _score_to_grade(64.9) == "C"


def test_grade_d():
    assert _score_to_grade(35) == "D"
    assert _score_to_grade(49.9) == "D"


def test_grade_f():
    assert _score_to_grade(0) == "F"
    assert _score_to_grade(34.9) == "F"


# ---------------------------------------------------------------------------
# Integration tests: compute_health_score
# ---------------------------------------------------------------------------

def test_all_lenses_perfect():
    """All lenses at 100 should give 100."""
    result = compute_health_score({
        "diversification": 100,
        "overlap": 100,
        "cost": 100,
        "risk": 100,
        "tax": 100,
    })
    assert result.score == 100.0
    assert result.grade == "A"
    assert result.lenses_available == 5
    assert result.lenses_total == 5


def test_all_lenses_zero():
    """All lenses at 0 should give 0."""
    result = compute_health_score({
        "diversification": 0,
        "overlap": 0,
        "cost": 0,
        "risk": 0,
        "tax": 0,
    })
    assert result.score == 0.0
    assert result.grade == "F"


def test_no_lenses():
    """No lenses available."""
    result = compute_health_score({})
    assert result.score == 0.0
    assert result.grade == "F"
    assert result.lenses_available == 0


def test_all_none():
    """All lenses None."""
    result = compute_health_score({
        "diversification": None,
        "overlap": None,
        "cost": None,
        "risk": None,
        "tax": None,
    })
    assert result.score == 0.0
    assert result.lenses_available == 0


def test_partial_lenses():
    """Only some lenses available — should re-weight."""
    result = compute_health_score({
        "diversification": 80,
        "cost": 60,
        "risk": 70,
        "overlap": None,
        "tax": None,
    })
    assert result.lenses_available == 3
    assert result.lenses_total == 5
    # Weights should sum to 1.0
    total_weight = sum(ls.weight for ls in result.lens_scores)
    assert abs(total_weight - 1.0) < 0.01
    # Score should be between 60 and 80
    assert 60 <= result.score <= 80


def test_overlap_missing():
    """Overlap often fails (Morningstar timeout) — should handle gracefully."""
    result = compute_health_score({
        "diversification": 70,
        "overlap": None,
        "cost": 80,
        "risk": 60,
        "tax": 50,
    })
    assert result.lenses_available == 4
    assert result.score > 0


def test_weights_sum_to_one():
    """Default weights must sum to 1.0."""
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001


def test_weights_proportional_reweighting():
    """When lenses are missing, remaining weights should maintain ratios."""
    # With only diversification (0.25) and risk (0.25), each should get 0.5
    result = compute_health_score({
        "diversification": 80,
        "risk": 60,
    })
    weights = {ls.lens: ls.weight for ls in result.lens_scores}
    assert abs(weights["diversification"] - 0.5) < 0.01
    assert abs(weights["risk"] - 0.5) < 0.01


def test_mixed_scores():
    """Typical real-world scenario with mixed scores."""
    result = compute_health_score({
        "diversification": 45,  # Concentrated
        "overlap": 75,  # Some overlap
        "cost": 30,  # Lots of regular plans
        "risk": 55,  # Moderate risk
        "tax": 70,  # Decent tax efficiency
    })
    # Weighted average: 45*0.25 + 75*0.20 + 30*0.15 + 55*0.25 + 70*0.15
    # = 11.25 + 15 + 4.5 + 13.75 + 10.5 = 55.0
    assert abs(result.score - 55.0) < 0.5
    assert result.grade == "C"


def test_lens_scores_sorted_by_weight():
    """Lens scores should be sorted by weight (highest first)."""
    result = compute_health_score({
        "diversification": 50,
        "overlap": 50,
        "cost": 50,
        "risk": 50,
        "tax": 50,
    })
    weights = [ls.weight for ls in result.lens_scores]
    assert weights == sorted(weights, reverse=True)


def test_summary_generated():
    """Summary should be a non-empty string."""
    result = compute_health_score({
        "diversification": 80,
        "overlap": 80,
        "cost": 80,
        "risk": 80,
        "tax": 80,
    })
    assert len(result.summary) > 0
    assert result.grade == "A"


def test_unknown_lens_ignored():
    """Unknown lens names should be ignored."""
    result = compute_health_score({
        "diversification": 80,
        "unknown_lens": 50,
    })
    assert result.lenses_available == 1
    assert len(result.lens_scores) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
