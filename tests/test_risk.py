"""Tests for the Risk Analyzer lens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.risk import (
    _holding_risk_level,
    _is_concentrated,
    analyze_risk,
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


def _make_portfolio(holdings: list[Holding]) -> Portfolio:
    return Portfolio(holdings=holdings, sources_used=["test"])


# ---------------------------------------------------------------------------
# Unit tests: risk level classification
# ---------------------------------------------------------------------------

def test_risk_level_equity():
    h = _make_holding(asset_type=AssetType.EQUITY)
    assert _holding_risk_level(h) == 5


def test_risk_level_liquid_fund():
    h = _make_holding(sector="Liquid Fund")
    assert _holding_risk_level(h) == 1


def test_risk_level_overnight_fund():
    h = _make_holding(sector="Overnight Fund")
    assert _holding_risk_level(h) == 1


def test_risk_level_large_cap():
    h = _make_holding(sector="Large Cap Fund")
    assert _holding_risk_level(h) == 4


def test_risk_level_small_cap():
    h = _make_holding(sector="Small Cap Fund")
    assert _holding_risk_level(h) == 5


def test_risk_level_mid_cap():
    h = _make_holding(sector="Mid Cap Fund")
    assert _holding_risk_level(h) == 5


def test_risk_level_balanced():
    h = _make_holding(sector="Balanced Advantage Fund")
    assert _holding_risk_level(h) == 4


def test_risk_level_gilt():
    h = _make_holding(sector="Gilt Fund")
    assert _holding_risk_level(h) == 3


def test_risk_level_gold_etf():
    h = _make_holding(sector="Gold ETF", asset_type=AssetType.ETF)
    assert _holding_risk_level(h) == 3


def test_risk_level_unknown_mf():
    h = _make_holding(sector="Some Unknown Category")
    assert _holding_risk_level(h) == 4  # default for unknown MF


def test_risk_level_etf_no_sector():
    h = _make_holding(asset_type=AssetType.ETF, sector=None)
    assert _holding_risk_level(h) == 4


# ---------------------------------------------------------------------------
# Integration tests: analyze_risk
# ---------------------------------------------------------------------------

def test_empty_portfolio():
    portfolio = _make_portfolio([])
    result = analyze_risk(portfolio)
    assert result.score == 50.0
    assert result.risk_label == "moderate"
    assert result.holdings_analyzed == 0


def test_single_holding():
    """Single holding = maximum concentration."""
    portfolio = _make_portfolio([
        _make_holding("SBI Large Cap", market_value=500000, sector="Large Cap Fund"),
    ])
    result = analyze_risk(portfolio)
    # D-013: a 100 % large-cap fund is a diversified single-manager exposure:
    # single-manager 20 + all-high 15 + no-low 10 = 45 penalty → 55. Still
    # penalised, but no longer scored as a single stock (was < 50; D-013's prose
    # estimate of 45 assumes the few-holdings penalty, which its rule gates off).
    assert result.score == 55.0
    assert result.risk_label in ("high", "very_high")
    assert result.top_n is not None
    assert len(result.concentrated_holdings) == 1
    assert result.concentrated_holdings[0].weight_pct == 100.0
    assert len(result.findings) > 0


def test_well_diversified_portfolio():
    """Many holdings, mix of asset types, no single holding > 10%."""
    holdings = [
        _make_holding(f"Large Cap {i}", market_value=50000, sector="Large Cap Fund")
        for i in range(5)
    ] + [
        _make_holding(f"Debt Fund {i}", market_value=30000, sector="Short Duration Fund")
        for i in range(3)
    ] + [
        _make_holding("Gold ETF", market_value=20000, sector="Gold ETF",
                       asset_type=AssetType.ETF),
        _make_holding("Liquid Fund", market_value=20000, sector="Liquid Fund"),
    ]
    portfolio = _make_portfolio(holdings)
    result = analyze_risk(portfolio)
    assert result.score >= 60
    # 65.8% in high-risk bucket (Large Cap = risk level 4) → "high" label
    assert result.risk_label == "high"


def test_concentrated_equity_portfolio():
    """All equity, one holding dominates."""
    portfolio = _make_portfolio([
        _make_holding("TCS", market_value=500000, asset_type=AssetType.EQUITY),
        _make_holding("Infosys", market_value=50000, asset_type=AssetType.EQUITY),
        _make_holding("Wipro", market_value=50000, asset_type=AssetType.EQUITY),
    ])
    result = analyze_risk(portfolio)
    # TCS is ~83% of portfolio, all high risk
    assert result.score < 40
    assert result.risk_label in ("high", "very_high")
    assert result.concentrated_holdings[0].scheme_name == "TCS"
    assert result.concentrated_holdings[0].weight_pct > 80


def test_top_n_concentration():
    """Test top-N metrics."""
    holdings = [
        _make_holding("Big Fund", market_value=700000),
        _make_holding("Medium Fund", market_value=200000),
        _make_holding("Small Fund", market_value=100000),
    ]
    portfolio = _make_portfolio(holdings)
    result = analyze_risk(portfolio)
    assert result.top_n is not None
    assert result.top_n.top_3_pct == 100.0  # only 3 holdings
    assert result.top_n.total_holdings == 3


def test_risk_buckets():
    """Test risk bucket distribution."""
    portfolio = _make_portfolio([
        _make_holding("Liquid", market_value=100000, sector="Liquid Fund"),
        _make_holding("Large Cap", market_value=100000, sector="Large Cap Fund"),
    ])
    result = analyze_risk(portfolio)
    assert result.risk_buckets is not None
    assert result.risk_buckets.low == 50.0
    assert result.risk_buckets.high == 50.0


def test_all_low_risk():
    """Portfolio entirely in low-risk instruments."""
    holdings = [
        _make_holding(f"Liquid {i}", market_value=50000, sector="Liquid Fund")
        for i in range(5)
    ] + [
        _make_holding(f"Overnight {i}", market_value=50000, sector="Overnight Fund")
        for i in range(5)
    ]
    portfolio = _make_portfolio(holdings)
    result = analyze_risk(portfolio)
    assert result.risk_buckets is not None
    assert result.risk_buckets.low == 100.0
    assert result.risk_buckets.high == 0.0
    assert result.score >= 70
    assert result.risk_label == "low"


def test_all_high_risk():
    """Portfolio entirely in high-risk instruments."""
    holdings = [
        _make_holding(f"Stock {i}", market_value=100000, asset_type=AssetType.EQUITY)
        for i in range(10)
    ]
    portfolio = _make_portfolio(holdings)
    result = analyze_risk(portfolio)
    assert result.risk_buckets is not None
    assert result.risk_buckets.high == 100.0
    assert result.score < 70


def test_holdings_with_no_value():
    """Holdings without market_value or invested_value are skipped."""
    portfolio = _make_portfolio([
        _make_holding("Fund A", market_value=None),
        _make_holding("Fund B", market_value=100000),
    ])
    result = analyze_risk(portfolio)
    # Only Fund B should be analyzed for concentration
    assert result.top_n is not None
    assert result.top_n.total_holdings == 1


def test_findings_single_holding_over_10pct():
    """Concentrated holding should generate a finding."""
    portfolio = _make_portfolio([
        _make_holding("Big Fund", market_value=500000),
        _make_holding("Small Fund", market_value=100000),
    ])
    result = analyze_risk(portfolio)
    assert any("over 10%" in f for f in result.findings)


def test_findings_no_low_risk():
    """All high-risk portfolio should mention missing low-risk."""
    holdings = [
        _make_holding(f"Equity {i}", market_value=50000, asset_type=AssetType.EQUITY)
        for i in range(5)
    ]
    portfolio = _make_portfolio(holdings)
    result = analyze_risk(portfolio)
    assert any("low-risk" in f.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# D-013: concentrated vs diversified instruments
# ---------------------------------------------------------------------------

def test_is_concentrated_classification():
    assert _is_concentrated(_make_holding(asset_type=AssetType.EQUITY))
    assert _is_concentrated(_make_holding(sector="Sectoral/Thematic"))
    assert _is_concentrated(_make_holding(sector=None))
    assert _is_concentrated(_make_holding(sector="Unknown", asset_type=AssetType.ETF))
    assert not _is_concentrated(_make_holding(sector="Index Funds"))
    assert not _is_concentrated(_make_holding(sector="Flexi Cap Fund"))
    assert not _is_concentrated(_make_holding(sector="Small Cap Fund"))
    assert not _is_concentrated(_make_holding(sector="Liquid Fund"))


def test_index_fund_portfolio_uses_single_manager_schedule():
    """60 % index + 30 % flexi + 10 % arbitrage: 20 (single manager) + 15 (all-high 90 %)."""
    portfolio = _make_portfolio([
        _make_holding("Nifty 50 Index", market_value=600000, sector="Index Funds"),
        _make_holding("Flexi Cap", market_value=300000, sector="Flexi Cap Fund"),
        _make_holding("Arbitrage", market_value=100000, sector="Arbitrage Fund"),
    ])
    result = analyze_risk(portfolio)
    assert result.score == 65.0
    assert result.risk_label == "high"


def test_small_cap_heavy_portfolio_is_very_high():
    """Level-5 share >= 50 % labels very_high even though funds are diversified."""
    portfolio = _make_portfolio([
        _make_holding("Small Cap", market_value=600000, sector="Small Cap Fund"),
        _make_holding("Liquid", market_value=400000, sector="Liquid Fund"),
    ])
    result = analyze_risk(portfolio)
    assert result.risk_label == "very_high"


def test_pure_direct_equity_keeps_bucket_penalty():
    """10 equal stocks: 15 + 15 (over-10 count) + 25 + 15 (buckets) + 10 (no low) → 20."""
    holdings = [
        _make_holding(f"Stock {i}", market_value=100000, asset_type=AssetType.EQUITY)
        for i in range(10)
    ]
    result = analyze_risk(_make_portfolio(holdings))
    assert result.score == 20.0
    assert result.risk_label == "very_high"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
