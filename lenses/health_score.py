"""Health Score — Composite portfolio health metric (0-100).

Combines sub-scores from all available analysis lenses into a single
number that represents overall portfolio health. The score is meaningful
at any level of completion — if some lenses are unavailable, the
remaining lenses are re-weighted proportionally.

Weights reflect relative importance:
- Diversification (25%): Asset allocation is the #1 driver of long-term returns
- Risk Profile (25%): Concentration risk can cause catastrophic losses
- Overlap (20%): Redundancy wastes expense ratios for identical exposure
- Cost Efficiency (15%): Expense drag compounds significantly over time
- Tax Awareness (15%): Tax efficiency preserves more of your returns

When Goal Alignment is available (6 lenses), use WEIGHTS_WITH_GOAL.
When only data-only lenses are available (5 lenses), use DEFAULT_WEIGHTS.
"""

from dataclasses import dataclass, field


# Lens name → default weight (must sum to 1.0)
# Used when only the 5 data-only lenses are available (no user input).
DEFAULT_WEIGHTS: dict[str, float] = {
    "diversification": 0.25,
    "risk": 0.25,
    "overlap": 0.20,
    "cost": 0.15,
    "tax": 0.15,
}

# Used when Goal Alignment (Phase 2) is also available (6 lenses).
# Applied by the frontend via client-side recomputation.
WEIGHTS_WITH_GOAL: dict[str, float] = {
    "diversification": 0.200,
    "goal_alignment": 0.200,
    "risk": 0.200,
    "overlap": 0.150,
    "cost": 0.125,
    "tax": 0.125,
}


@dataclass
class LensScore:
    """A single lens contribution to the health score."""
    lens: str  # "diversification" | "overlap" | "cost" | "risk" | "tax"
    label: str  # Human-readable name
    score: float  # 0-100 sub-score
    weight: float  # Effective weight after re-weighting (0-1)
    weighted_score: float  # score * weight


@dataclass
class HealthScoreResult:
    """Full health score output."""
    score: float  # 0-100 composite
    grade: str  # "A" | "B" | "C" | "D" | "F"
    lenses_available: int
    lenses_total: int  # Always 5 for now
    lens_scores: list[LensScore] = field(default_factory=list)
    summary: str = ""  # One-line plain-English summary


_LENS_LABELS: dict[str, str] = {
    "diversification": "Diversification",
    "goal_alignment": "Goal Alignment",
    "overlap": "Overlap",
    "cost": "Cost Efficiency",
    "risk": "Risk Profile",
    "tax": "Tax Awareness",
}


def compute_health_score(
    scores: dict[str, float | None],
) -> HealthScoreResult:
    """Compute the composite health score from available lens sub-scores.

    Args:
        scores: Dict mapping lens name → sub-score (0-100) or None if unavailable.
                Expected keys: diversification, overlap, cost, risk, tax

    Returns:
        HealthScoreResult with composite score, grade, and per-lens breakdown.
    """
    # Filter to available lenses
    available: dict[str, float] = {
        k: v for k, v in scores.items()
        if v is not None and k in DEFAULT_WEIGHTS
    }

    if not available:
        return HealthScoreResult(
            score=0.0,
            grade="F",
            lenses_available=0,
            lenses_total=len(DEFAULT_WEIGHTS),
            summary="No analysis data available.",
        )

    # Re-weight proportionally based on available lenses
    total_default_weight = sum(DEFAULT_WEIGHTS[k] for k in available)
    effective_weights: dict[str, float] = {
        k: DEFAULT_WEIGHTS[k] / total_default_weight
        for k in available
    }

    # Compute weighted composite
    lens_scores: list[LensScore] = []
    composite = 0.0

    for lens, sub_score in available.items():
        weight = effective_weights[lens]
        weighted = sub_score * weight
        composite += weighted
        lens_scores.append(LensScore(
            lens=lens,
            label=_LENS_LABELS.get(lens, lens),
            score=round(sub_score, 1),
            weight=round(weight, 3),
            weighted_score=round(weighted, 1),
        ))

    # Sort by weight (highest first)
    lens_scores.sort(key=lambda ls: ls.weight, reverse=True)

    composite = round(composite, 1)
    grade = _score_to_grade(composite)
    summary = _generate_summary(composite, grade, len(available))

    return HealthScoreResult(
        score=composite,
        grade=grade,
        lenses_available=len(available),
        lenses_total=len(DEFAULT_WEIGHTS),
        lens_scores=lens_scores,
        summary=summary,
    )


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


def _generate_summary(score: float, grade: str, lenses: int) -> str:
    """Generate a one-line summary of the health score."""
    if score >= 80:
        return f"Your portfolio scores {round(score)}/100 (Grade {grade}) across {lenses} dimensions analyzed."
    if score >= 65:
        return f"Your portfolio scores {round(score)}/100 (Grade {grade}) across {lenses} dimensions analyzed."
    if score >= 50:
        return f"Your portfolio scores {round(score)}/100 (Grade {grade}) across {lenses} dimensions analyzed."
    if score >= 35:
        return f"Your portfolio scores {round(score)}/100 (Grade {grade}) across {lenses} dimensions analyzed."
    return f"Your portfolio scores {round(score)}/100 (Grade {grade}) across {lenses} dimensions analyzed."
