"""Rupee-Cost Aggregator — computes total ₹ cost of portfolio inefficiencies.

Aggregates rupee-denominated impact figures from existing analysis lenses:
1. Cost lens: regular→direct plan expense ratio savings (annual)
2. Overlap lens: duplicate expense ratios on overlapping fund pairs (annual)
3. Tax lens: worst-case tax liability if all holdings sold today (one-time)

All computation is deterministic. No LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from lenses.cost import CostResult, _EXPENSE_ESTIMATES, _FALLBACK_ESTIMATES
from lenses.overlap import OverlapResult
from lenses.tax import TaxResult
from lenses.models import AssetType, Holding, Portfolio


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class RupeeCostSummary:
    """Aggregated rupee cost of portfolio inefficiencies."""
    total_annual_drag: float      # cost_drag + overlap_drag (recurring annual)
    cost_drag: float              # annual savings if regular → direct
    overlap_drag: float           # annual duplicate ER cost from overlapping funds
    tax_liability: float          # worst-case one-time tax if sold today
    disclaimer: str = (
        "Estimates based on current holdings and published expense ratios. "
        "Actual costs vary. Tax figures assume worst-case (short-term) rates."
    )


# ---------------------------------------------------------------------------
# Overlap drag calculation
# ---------------------------------------------------------------------------

def _get_expense_ratio(holding: Holding) -> float:
    """Get the applicable expense ratio for a holding (as percentage, e.g. 1.60)."""
    cat = holding.sector or ""

    if cat in _EXPENSE_ESTIMATES:
        reg, direct = _EXPENSE_ESTIMATES[cat]
        # Use the plan type the user actually holds
        name_upper = (holding.scheme_name or "").upper()
        if "DIRECT" in name_upper or "- DIR" in name_upper:
            return direct
        return reg

    # Fallback by asset type
    if holding.asset_type == AssetType.ETF:
        return 0.20
    if holding.asset_type == AssetType.EQUITY:
        return 0.0

    cat_upper = cat.upper()
    if any(kw in cat_upper for kw in ("DEBT", "BOND", "GILT", "LIQUID", "MONEY")):
        reg, _ = _FALLBACK_ESTIMATES["debt"]
        return reg
    if any(kw in cat_upper for kw in ("HYBRID", "BALANCED", "ARBITRAGE")):
        reg, _ = _FALLBACK_ESTIMATES["hybrid"]
        return reg

    reg, _ = _FALLBACK_ESTIMATES["equity"]
    return reg


def _compute_overlap_drag(
    overlap_result: OverlapResult,
    portfolio: Portfolio,
) -> float:
    """Compute annual ₹ cost of duplicate expense ratios from overlapping funds.

    For each fund pair with overlap > 25%:
    - Take the smaller fund's AUM
    - Multiply by overlap percentage (the fraction of holdings that are duplicated)
    - Multiply by the average expense ratio of the pair
    - This estimates the annual cost of paying two ERs for the same stocks
    """
    if not overlap_result or not overlap_result.fund_pairs:
        return 0.0

    # Build ISIN → holding lookup
    isin_to_holding: dict[str, Holding] = {}
    for h in portfolio.holdings:
        if h.isin:
            isin_to_holding[h.isin] = h

    total_drag = 0.0

    for pair in overlap_result.fund_pairs:
        if pair.overlap_pct < 25:
            continue

        holding_a = isin_to_holding.get(pair.fund_a_isin)
        holding_b = isin_to_holding.get(pair.fund_b_isin)

        if not holding_a or not holding_b:
            continue

        aum_a = holding_a.market_value or holding_a.invested_value or 0
        aum_b = holding_b.market_value or holding_b.invested_value or 0

        if aum_a <= 0 or aum_b <= 0:
            continue

        smaller_aum = min(aum_a, aum_b)
        er_a = _get_expense_ratio(holding_a)
        er_b = _get_expense_ratio(holding_b)
        avg_er = (er_a + er_b) / 2

        # Overlap drag: smaller fund's AUM × overlap fraction × average ER
        drag = smaller_aum * (pair.overlap_pct / 100) * (avg_er / 100)
        total_drag += drag

    return round(total_drag, 2)


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def compute_rupee_cost(
    cost_result: CostResult | None,
    overlap_result: OverlapResult | None,
    tax_result: TaxResult | None,
    portfolio: Portfolio,
) -> RupeeCostSummary:
    """Aggregate rupee-cost figures from all available lenses."""

    # Cost drag: annual savings from switching regular → direct plans
    cost_drag = 0.0
    if cost_result and cost_result.summary:
        cost_drag = cost_result.summary.total_potential_savings

    # Overlap drag: annual duplicate ER cost
    overlap_drag = 0.0
    if overlap_result:
        overlap_drag = _compute_overlap_drag(overlap_result, portfolio)

    # Tax liability: worst-case one-time (not annual)
    tax_liability = 0.0
    if tax_result and tax_result.summary:
        tax_liability = tax_result.summary.estimated_total_tax

    total_annual_drag = round(cost_drag + overlap_drag, 2)

    return RupeeCostSummary(
        total_annual_drag=total_annual_drag,
        cost_drag=round(cost_drag, 2),
        overlap_drag=overlap_drag,
        tax_liability=round(tax_liability, 2),
    )
