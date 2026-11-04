"""Cost Efficiency Analyzer — Phase 1 Lens (3c in roadmap, Lens 3 in product spec).

Detects regular-plan MF holdings where a direct plan exists, and estimates
the annual cost savings from switching. Also flags high-expense holdings.

Current data source: Estimated expense ratio differentials by fund category.
TODO: Add exact expense ratios from AMC factsheets or AMFI data when available.

All computation is deterministic. No LLM calls.
"""

import logging
from dataclasses import dataclass, field

from lenses.scheme_master import AMFIMaster, SchemeInfo
from lenses.scheme_master import classify_direct_regular, find_direct_alternative
from lenses.models import AssetType, Holding, Portfolio

logger = logging.getLogger(__name__)


def _classify_plan_from_name(scheme_name: str) -> str | None:
    """Classify direct/regular from scheme name keywords (no AMFI master needed)."""
    name_upper = scheme_name.upper()
    if "DIRECT" in name_upper or "- DIR" in name_upper:
        return "direct"
    if "REGULAR" in name_upper or "- REG" in name_upper:
        return "regular"
    return None


# ---------------------------------------------------------------------------
# Estimated expense ratio differentials by fund category (Indian MFs)
# These are industry-typical ranges. TODO: replace with exact data.
# ---------------------------------------------------------------------------

# (regular_pct, direct_pct) — annual expense ratios
_EXPENSE_ESTIMATES: dict[str, tuple[float, float]] = {
    # Equity
    "Large Cap Fund": (1.60, 0.55),
    "Mid Cap Fund": (1.80, 0.65),
    "Small Cap Fund": (1.85, 0.70),
    "Large & Mid Cap Fund": (1.70, 0.60),
    "Multi Cap Fund": (1.75, 0.60),
    "Flexi Cap Fund": (1.70, 0.55),
    "ELSS": (1.75, 0.65),
    "Sectoral/Thematic": (1.80, 0.65),
    "Sectoral/ Thematic": (1.80, 0.65),  # AMFI double space variant
    "Value Fund": (1.70, 0.55),
    "Contra Fund": (1.70, 0.55),
    "Dividend Yield Fund": (1.70, 0.55),
    "Focused Fund": (1.75, 0.60),
    "Index Funds": (0.40, 0.15),
    # Debt
    "Liquid Fund": (0.30, 0.10),
    "Ultra Short Duration Fund": (0.60, 0.20),
    "Low Duration Fund": (0.70, 0.25),
    "Short Duration Fund": (0.80, 0.30),
    "Medium Duration Fund": (0.90, 0.35),
    "Medium to Long Duration Fund": (1.00, 0.40),
    "Long Duration Fund": (0.80, 0.30),
    "Dynamic Bond": (1.00, 0.40),
    "Corporate Bond Fund": (0.70, 0.25),
    "Credit Risk Fund": (1.20, 0.50),
    "Banking and PSU Fund": (0.60, 0.20),
    "Gilt Fund": (0.70, 0.25),
    "Gilt Fund with 10 year constant duration": (0.70, 0.25),
    "Money Market Fund": (0.30, 0.10),
    "Overnight Fund": (0.20, 0.05),
    "Floater Fund": (0.60, 0.20),
    # Hybrid
    "Balanced Advantage Fund": (1.50, 0.50),
    "Aggressive Hybrid Fund": (1.60, 0.55),
    "Conservative Hybrid Fund": (1.20, 0.40),
    "Multi Asset Allocation": (1.50, 0.50),
    "Arbitrage Fund": (0.80, 0.30),
    "Equity Savings": (1.20, 0.40),
    # FoF / International
    "Fund of Funds (Overseas)": (1.50, 0.50),
    "Fund of Funds (Domestic)": (1.00, 0.30),
    # ETFs — no regular/direct distinction
    "Gold ETF": (0.50, 0.50),
    "Other  ETFs": (0.10, 0.10),
}

# Fallback by broad asset class
_FALLBACK_ESTIMATES: dict[str, tuple[float, float]] = {
    "equity": (1.75, 0.60),
    "debt": (0.70, 0.25),
    "hybrid": (1.40, 0.50),
}


def _get_expense_estimate(holding: Holding) -> tuple[float, float]:
    """Get estimated (regular, direct) expense ratios for a holding."""
    cat = holding.sector or ""
    if cat in _EXPENSE_ESTIMATES:
        return _EXPENSE_ESTIMATES[cat]

    # Fallback by asset type
    if holding.asset_type == AssetType.ETF:
        return (0.20, 0.20)  # ETFs have low, similar expense ratios
    if holding.asset_type == AssetType.EQUITY:
        return (0.0, 0.0)  # Direct stocks have no expense ratio

    # Check category keywords for broad classification
    cat_upper = cat.upper()
    if any(kw in cat_upper for kw in ("DEBT", "BOND", "GILT", "LIQUID", "MONEY")):
        return _FALLBACK_ESTIMATES["debt"]
    if any(kw in cat_upper for kw in ("HYBRID", "BALANCED", "ARBITRAGE")):
        return _FALLBACK_ESTIMATES["hybrid"]

    return _FALLBACK_ESTIMATES["equity"]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class RegularPlanFinding:
    """A regular-plan holding where a direct plan exists."""
    scheme_name: str
    isin: str | None
    plan_type: str  # "regular"
    market_value: float
    estimated_regular_expense_pct: float
    estimated_direct_expense_pct: float
    estimated_annual_savings: float  # INR saved per year by switching
    direct_alternative_name: str | None = None
    direct_alternative_isin: str | None = None
    category: str | None = None


@dataclass
class CostSummary:
    """Summary of portfolio cost analysis."""
    total_annual_cost: float  # estimated total expense drag (INR/year)
    total_potential_savings: float  # savings if all regular → direct (INR/year)
    regular_plan_count: int
    direct_plan_count: int
    unclassified_count: int  # couldn't determine plan type
    avg_expense_ratio_pct: float  # weighted average expense ratio


@dataclass
class CostResult:
    """Full output of the Cost Efficiency Analyzer lens."""
    score: float  # 0-100, higher = more cost efficient

    summary: CostSummary
    regular_plan_findings: list[RegularPlanFinding] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    holdings_analyzed: int = 0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_cost(portfolio: Portfolio, amfi_master: AMFIMaster | None) -> CostResult:
    """Run the cost efficiency analysis on a portfolio.

    Phase 1 lens — deterministic, no user input needed.
    """
    # Only analyze mutual funds for direct/regular plan detection.
    # ETFs don't have a meaningful direct/regular distinction —
    # they trade on exchanges and expense ratios are fixed.
    mf_etf_holdings = [
        h for h in portfolio.holdings
        if h.asset_type == AssetType.MUTUAL_FUND
    ]

    if not mf_etf_holdings:
        return CostResult(
            score=100.0,
            summary=CostSummary(
                total_annual_cost=0, total_potential_savings=0,
                regular_plan_count=0, direct_plan_count=0,
                unclassified_count=0, avg_expense_ratio_pct=0,
            ),
            holdings_analyzed=0,
        )

    regular_findings: list[RegularPlanFinding] = []
    regular_count = 0
    direct_count = 0
    unclassified_count = 0
    total_annual_cost = 0.0
    total_potential_savings = 0.0
    total_weighted_expense = 0.0
    total_value = 0.0

    for h in mf_etf_holdings:
        value = h.market_value or h.invested_value or 0
        if amfi_master:
            plan_type = classify_direct_regular(h, amfi_master)
        else:
            # Fallback: classify from scheme name keywords only
            plan_type = _classify_plan_from_name(h.scheme_name)
        reg_exp, dir_exp = _get_expense_estimate(h)

        if plan_type == "regular":
            regular_count += 1
            annual_cost = value * reg_exp / 100
            annual_savings = value * (reg_exp - dir_exp) / 100
            total_annual_cost += annual_cost
            total_potential_savings += annual_savings
            total_weighted_expense += value * reg_exp

            # Find direct alternative
            alt_name = None
            alt_isin = None
            if amfi_master:
                alt = find_direct_alternative(h, amfi_master)
                if alt:
                    alt_name = alt.scheme_name
                    alt_isin = alt.isin_growth

            regular_findings.append(RegularPlanFinding(
                scheme_name=h.scheme_name,
                isin=h.isin,
                plan_type="regular",
                market_value=value,
                estimated_regular_expense_pct=reg_exp,
                estimated_direct_expense_pct=dir_exp,
                estimated_annual_savings=round(annual_savings, 2),
                direct_alternative_name=alt_name,
                direct_alternative_isin=alt_isin,
                category=h.sector,
            ))

        elif plan_type == "direct":
            direct_count += 1
            annual_cost = value * dir_exp / 100
            total_annual_cost += annual_cost
            total_weighted_expense += value * dir_exp

        else:
            unclassified_count += 1
            # Assume midpoint for unclassified
            mid_exp = (reg_exp + dir_exp) / 2
            total_annual_cost += value * mid_exp / 100
            total_weighted_expense += value * mid_exp

        total_value += value

    # Sort findings by savings (highest first)
    regular_findings.sort(key=lambda f: f.estimated_annual_savings, reverse=True)

    avg_expense = (total_weighted_expense / total_value * 100) if total_value > 0 else 0

    summary = CostSummary(
        total_annual_cost=round(total_annual_cost, 2),
        total_potential_savings=round(total_potential_savings, 2),
        regular_plan_count=regular_count,
        direct_plan_count=direct_count,
        unclassified_count=unclassified_count,
        avg_expense_ratio_pct=round(avg_expense / 100, 2),  # as percentage
    )

    findings = _generate_findings(summary, regular_findings)
    score = _compute_cost_score(summary, mf_etf_holdings)

    return CostResult(
        score=score,
        summary=summary,
        regular_plan_findings=regular_findings,
        findings=findings,
        holdings_analyzed=len(mf_etf_holdings),
    )


def _compute_cost_score(summary: CostSummary, holdings: list[Holding]) -> float:
    """Compute cost efficiency score (0-100, higher = lower costs = better).

    Penalizes for:
    - High proportion of regular plans (main driver)
    - High average expense ratio
    """
    total_mf_etf = summary.regular_plan_count + summary.direct_plan_count + summary.unclassified_count
    if total_mf_etf == 0:
        return 100.0

    # Regular plan ratio penalty (0-60 points)
    regular_ratio = summary.regular_plan_count / total_mf_etf
    plan_penalty = regular_ratio * 60

    # Average expense ratio penalty (0-30 points)
    # Benchmark: 0.5% is excellent, 2% is terrible
    avg_exp = summary.avg_expense_ratio_pct
    if avg_exp <= 0.5:
        expense_penalty = 0
    elif avg_exp >= 2.0:
        expense_penalty = 30
    else:
        expense_penalty = (avg_exp - 0.5) / 1.5 * 30

    # Savings potential penalty (0-10 points)
    total_value = sum(h.market_value or h.invested_value or 0 for h in holdings)
    if total_value > 0:
        savings_ratio = summary.total_potential_savings / total_value
        savings_penalty = min(savings_ratio * 100 * 10, 10)  # 1% savings = 10 pts
    else:
        savings_penalty = 0

    total_penalty = min(plan_penalty + expense_penalty + savings_penalty, 100)
    return round(max(100 - total_penalty, 0), 1)


def _generate_findings(
    summary: CostSummary,
    regular_findings: list[RegularPlanFinding],
) -> list[str]:
    """Generate plain-English findings about cost efficiency."""
    findings: list[str] = []

    if summary.regular_plan_count > 0:
        findings.append(
            f"{summary.regular_plan_count} of your mutual fund holdings are "
            f"regular plans. Direct plan equivalents have lower expense ratios — "
            f"estimated difference: ₹{summary.total_potential_savings:,.0f} per year."
        )

    # Top savings opportunities
    for rf in regular_findings[:3]:
        if rf.estimated_annual_savings >= 100:
            alt_text = ""
            if rf.direct_alternative_name:
                alt_text = f" The direct plan alternative is '{rf.direct_alternative_name}'."
            findings.append(
                f"'{rf.scheme_name}' is a regular plan "
                f"(estimated expense ratio: {rf.estimated_regular_expense_pct:.1f}%). "
                f"Direct plan equivalent has ~₹{rf.estimated_annual_savings:,.0f}/year lower fees.{alt_text}"
            )

    if summary.regular_plan_count == 0 and summary.direct_plan_count > 0:
        findings.append(
            "All your mutual fund holdings are direct plans — "
            "you're already minimizing expense ratio costs."
        )

    if summary.avg_expense_ratio_pct >= 1.5:
        findings.append(
            f"Your weighted average expense ratio is "
            f"{summary.avg_expense_ratio_pct:.2f}%, which is above the "
            f"category average."
        )

    return findings
