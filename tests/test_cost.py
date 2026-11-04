"""Tests for the Cost Efficiency Analyzer lens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.cost import (
    CostResult,
    _get_expense_estimate,
    analyze_cost,
)
from lenses.models import AssetType, Holding, Portfolio


def _make_holding(
    name: str = "Test Fund",
    isin: str | None = "INF000000001",
    asset_type: AssetType = AssetType.MUTUAL_FUND,
    sector: str | None = None,
    market_value: float | None = 100000.0,
    units: float = 100.0,
    source: str = "test",
) -> Holding:
    return Holding(
        scheme_name=name, isin=isin, asset_type=asset_type, sector=sector,
        market_value=market_value, units=units, source=source,
    )


# ---------------------------------------------------------------------------
# Unit tests: expense ratio estimates
# ---------------------------------------------------------------------------

def test_expense_estimate_large_cap():
    h = _make_holding(sector="Large Cap Fund")
    reg, direct = _get_expense_estimate(h)
    assert reg > direct
    assert reg > 1.0  # Regular should be >1%
    assert direct < 1.0  # Direct should be <1%


def test_expense_estimate_liquid():
    h = _make_holding(sector="Liquid Fund")
    reg, direct = _get_expense_estimate(h)
    assert reg < 0.5  # Liquid funds have low expense ratios
    assert direct < 0.2


def test_expense_estimate_etf():
    h = _make_holding(asset_type=AssetType.ETF, sector="Other  ETFs")
    reg, direct = _get_expense_estimate(h)
    assert reg <= 0.2  # ETFs have minimal expense ratios


def test_expense_estimate_equity():
    h = _make_holding(asset_type=AssetType.EQUITY)
    reg, direct = _get_expense_estimate(h)
    assert reg == 0 and direct == 0  # Stocks have no expense ratio


def test_expense_estimate_unknown_category():
    h = _make_holding(sector="Some Unknown Category")
    reg, direct = _get_expense_estimate(h)
    assert reg > 0  # Should fall back to equity defaults


# ---------------------------------------------------------------------------
# Integration tests: full analysis (without AMFI master)
# ---------------------------------------------------------------------------

def test_empty_portfolio():
    portfolio = Portfolio(holdings=[])
    result = analyze_cost(portfolio, None)
    assert result.score == 100.0
    assert result.holdings_analyzed == 0


def test_equity_only_portfolio():
    """Direct stocks only → score 100 (no expense ratios)."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="TCS", asset_type=AssetType.EQUITY, market_value=100000),
    ])
    result = analyze_cost(portfolio, None)
    assert result.score == 100.0
    assert result.holdings_analyzed == 0  # equities not analyzed


def test_direct_plan_portfolio():
    """All direct plans → high score."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A Direct Growth", sector="Large Cap Fund", market_value=200000),
        _make_holding(name="Fund B Direct Growth", sector="Mid Cap Fund", market_value=100000),
    ])
    # Without AMFI master, classification falls back to name keywords
    result = analyze_cost(portfolio, None)
    assert result.summary.direct_plan_count == 2
    assert result.summary.regular_plan_count == 0
    assert result.score > 80


def test_regular_plan_portfolio():
    """All regular plans → low score, findings generated."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A Regular Growth", sector="Large Cap Fund", market_value=500000),
        _make_holding(name="Fund B Regular Growth", sector="Mid Cap Fund", market_value=300000),
    ])
    result = analyze_cost(portfolio, None)
    assert result.summary.regular_plan_count == 2
    assert result.summary.total_potential_savings > 0
    assert len(result.regular_plan_findings) == 2
    assert result.score < 50
    assert len(result.findings) > 0


def test_mixed_plan_portfolio():
    """Mix of direct and regular → medium score."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A Direct Growth", sector="Large Cap Fund", market_value=300000),
        _make_holding(name="Fund B Regular Growth", sector="Mid Cap Fund", market_value=200000),
    ])
    result = analyze_cost(portfolio, None)
    assert result.summary.regular_plan_count == 1
    assert result.summary.direct_plan_count == 1
    assert 20 < result.score < 80


def test_savings_calculation():
    """Verify savings math is correct."""
    portfolio = Portfolio(holdings=[
        _make_holding(
            name="SBI Bluechip Regular Growth",
            sector="Large Cap Fund",
            market_value=1000000,
        ),
    ])
    result = analyze_cost(portfolio, None)
    assert result.summary.regular_plan_count == 1
    finding = result.regular_plan_findings[0]
    # Savings should be market_value × (regular - direct) / 100
    expected_savings = 1000000 * (finding.estimated_regular_expense_pct - finding.estimated_direct_expense_pct) / 100
    assert abs(finding.estimated_annual_savings - expected_savings) < 0.01


def test_findings_sorted_by_savings():
    """Regular plan findings should be sorted by savings (highest first)."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Small Fund Regular Growth", sector="Large Cap Fund", market_value=50000),
        _make_holding(name="Big Fund Regular Growth", sector="Large Cap Fund", market_value=500000),
        _make_holding(name="Med Fund Regular Growth", sector="Large Cap Fund", market_value=200000),
    ])
    result = analyze_cost(portfolio, None)
    savings = [f.estimated_annual_savings for f in result.regular_plan_findings]
    assert savings == sorted(savings, reverse=True)


def test_unclassified_plans():
    """Holdings without 'Direct' or 'Regular' in name → unclassified."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Parag Parikh Flexi Cap Fund Growth", sector="Flexi Cap Fund"),
    ])
    result = analyze_cost(portfolio, None)
    assert result.summary.unclassified_count == 1


def test_total_annual_cost():
    """Verify total annual cost is computed."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund Regular Growth", sector="Large Cap Fund", market_value=100000),
    ])
    result = analyze_cost(portfolio, None)
    assert result.summary.total_annual_cost > 0
    # Should be market_value × regular_expense / 100
    reg_exp = result.regular_plan_findings[0].estimated_regular_expense_pct
    expected = 100000 * reg_exp / 100
    assert abs(result.summary.total_annual_cost - expected) < 1


def test_findings_message_savings():
    """Findings should mention cost difference for regular plans."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund Regular Growth", sector="Large Cap Fund", market_value=500000),
    ])
    result = analyze_cost(portfolio, None)
    assert any("lower" in f.lower() or "cheaper" in f.lower() or "difference" in f.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {test_fn.__name__}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
