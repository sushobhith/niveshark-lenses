"""Tests for the Diversification Analyzer lens."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.diversification import (
    AssetClass,
    MarketCap,
    _classify_asset_class,
    _classify_market_cap,
    _compute_hhi,
    _hhi_to_score,
    _market_cap_score,
    _weights_from_values,
    analyze_diversification,
)
from lenses.models import AssetType, Holding, Portfolio


def _make_holding(
    name: str = "Test Fund",
    asset_type: AssetType = AssetType.MUTUAL_FUND,
    sector: str | None = None,
    market_value: float | None = 100000.0,
    invested_value: float | None = 90000.0,
    units: float = 100.0,
    source: str = "test",
) -> Holding:
    return Holding(
        scheme_name=name,
        asset_type=asset_type,
        sector=sector,
        market_value=market_value,
        invested_value=invested_value,
        units=units,
        source=source,
    )


# ---------------------------------------------------------------------------
# Unit tests: classification
# ---------------------------------------------------------------------------

def test_classify_equity_stock():
    h = _make_holding(asset_type=AssetType.EQUITY)
    assert _classify_asset_class(h) == AssetClass.EQUITY


def test_classify_large_cap_fund():
    h = _make_holding(sector="Large Cap Fund")
    assert _classify_asset_class(h) == AssetClass.EQUITY
    assert _classify_market_cap(h) == MarketCap.LARGE


def test_classify_mid_cap_fund():
    h = _make_holding(sector="Mid Cap Fund")
    assert _classify_asset_class(h) == AssetClass.EQUITY
    assert _classify_market_cap(h) == MarketCap.MID


def test_classify_small_cap_fund():
    h = _make_holding(sector="Small Cap Fund")
    assert _classify_asset_class(h) == AssetClass.EQUITY
    assert _classify_market_cap(h) == MarketCap.SMALL


def test_classify_flexi_cap():
    h = _make_holding(sector="Flexi Cap Fund")
    assert _classify_asset_class(h) == AssetClass.EQUITY
    assert _classify_market_cap(h) == MarketCap.MULTI_FLEXI


def test_classify_debt_fund():
    h = _make_holding(sector="Liquid Fund")
    assert _classify_asset_class(h) == AssetClass.DEBT
    assert _classify_market_cap(h) is None  # not equity


def test_classify_gold_etf():
    h = _make_holding(sector="Gold ETF", asset_type=AssetType.ETF)
    assert _classify_asset_class(h) == AssetClass.GOLD


def test_classify_hybrid():
    h = _make_holding(sector="Balanced Advantage Fund")
    assert _classify_asset_class(h) == AssetClass.HYBRID


def test_classify_international():
    h = _make_holding(sector="Fund of Funds (Overseas)")
    assert _classify_asset_class(h) == AssetClass.INTERNATIONAL


def test_classify_elss():
    h = _make_holding(sector="ELSS")
    assert _classify_asset_class(h) == AssetClass.EQUITY
    # D-014: ELSS has no cap mandate and is run as a flexi-cap strategy (was UNCLASSIFIED)
    assert _classify_market_cap(h) == MarketCap.MULTI_FLEXI


def test_classify_index_funds_by_name():
    """D-014: index funds / Other ETFs are classified from the index in the name."""
    cases = {
        "UTI Nifty 50 Index Fund - Direct Plan - Growth": MarketCap.LARGE,
        "UTI Nifty Next 50 Index Fund - Direct Plan - Growth": MarketCap.LARGE,
        "HDFC Index Fund-S&P BSE Sensex Plan": MarketCap.LARGE,
        "ICICI Prudential Nifty 100 Index Fund": MarketCap.LARGE,
        "Nippon India ETF Junior BeES": MarketCap.LARGE,
        "Motilal Oswal Nifty Midcap 150 Index Fund": MarketCap.MID,
        "Nippon India Nifty Smallcap 250 Index Fund": MarketCap.SMALL,
        "Motilal Oswal Nifty 500 Index Fund": MarketCap.MULTI_FLEXI,
        "Navi Nifty India Total Market Index Fund": MarketCap.MULTI_FLEXI,
        "Some Nifty Bank Index Fund": MarketCap.UNCLASSIFIED,
    }
    for name, expected in cases.items():
        h = _make_holding(name=name, sector="Index Funds")
        assert _classify_market_cap(h) == expected, name
    etf = _make_holding(name="SBI Nifty 50 ETF", sector="Other  ETFs", asset_type=AssetType.ETF)
    assert _classify_market_cap(etf) == MarketCap.LARGE


def test_market_cap_score_anchors():
    """D-014: distance from the 70/20/10 market mix."""
    assert _market_cap_score({MarketCap.LARGE: 100.0}) == 70.0
    assert _market_cap_score({MarketCap.MULTI_FLEXI: 100.0}) == 100.0
    assert _market_cap_score({MarketCap.SMALL: 100.0}) == 10.0
    assert _market_cap_score({MarketCap.LARGE_MID: 100.0}) == 70.0
    # Unclassified is excluded from the vector; alone it scores 0
    assert _market_cap_score({MarketCap.LARGE: 50.0, MarketCap.UNCLASSIFIED: 50.0}) == 70.0
    assert _market_cap_score({MarketCap.UNCLASSIFIED: 100.0}) == 0.0


def test_classify_no_sector_mf():
    """MF without sector defaults to equity."""
    h = _make_holding(sector=None)
    assert _classify_asset_class(h) == AssetClass.EQUITY


def test_classify_debt_by_name_fallback():
    """Falls back to name keywords when no AMFI category."""
    h = _make_holding(name="HDFC Liquid Fund Direct Growth", sector=None)
    assert _classify_asset_class(h) == AssetClass.DEBT


# ---------------------------------------------------------------------------
# Unit tests: HHI and scoring
# ---------------------------------------------------------------------------

def test_hhi_single_bucket():
    """100% in one bucket → HHI = 10000."""
    assert _compute_hhi({"equity": 1.0}) == 10000.0


def test_hhi_equal_two():
    """50/50 split → HHI = 5000."""
    assert _compute_hhi({"equity": 0.5, "debt": 0.5}) == 5000.0


def test_hhi_equal_four():
    """25% each → HHI = 2500."""
    hhi = _compute_hhi({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    assert abs(hhi - 2500.0) < 0.01


def test_hhi_to_score_perfect():
    """Perfect equal distribution → score 100."""
    assert _hhi_to_score(2000, max_categories=5) == 100.0


def test_hhi_to_score_concentrated():
    """Single bucket → score 0."""
    assert _hhi_to_score(10000, max_categories=5) == 0.0


def test_hhi_to_score_single_category():
    """Only 1 possible category → score 0 (can't be diversified)."""
    assert _hhi_to_score(10000, max_categories=1) == 0.0


def test_weights_from_values():
    weights = _weights_from_values({"a": 300, "b": 700})
    assert abs(weights["a"] - 0.3) < 0.001
    assert abs(weights["b"] - 0.7) < 0.001


def test_weights_from_values_empty():
    assert _weights_from_values({}) == {}
    assert _weights_from_values({"a": 0}) == {}


# ---------------------------------------------------------------------------
# Integration tests: full analysis
# ---------------------------------------------------------------------------

def test_empty_portfolio():
    portfolio = Portfolio(holdings=[])
    result = analyze_diversification(portfolio)
    assert result.score == 0.0
    assert result.holdings_analyzed == 0


def test_single_equity_holding():
    portfolio = Portfolio(holdings=[
        _make_holding(sector="Large Cap Fund", market_value=100000),
    ])
    result = analyze_diversification(portfolio)
    assert result.holdings_analyzed == 1
    assert result.holdings_with_value == 1
    assert result.asset_class.weights[AssetClass.EQUITY] == 1.0
    # Single holding → low score
    assert result.score < 30


def test_diversified_portfolio():
    """A well-diversified portfolio should score higher."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Large Cap Fund A", sector="Large Cap Fund", market_value=250000),
        _make_holding(name="Mid Cap Fund A", sector="Mid Cap Fund", market_value=150000),
        _make_holding(name="Small Cap Fund A", sector="Small Cap Fund", market_value=100000),
        _make_holding(name="Liquid Fund A", sector="Liquid Fund", market_value=200000),
        _make_holding(name="Gold ETF", sector="Gold ETF", asset_type=AssetType.ETF, market_value=100000),
        _make_holding(name="Intl FoF", sector="Fund of Funds (Overseas)", market_value=100000),
        _make_holding(name="Hybrid Fund", sector="Balanced Advantage Fund", market_value=100000),
    ])
    result = analyze_diversification(portfolio)
    assert result.score > 60
    assert len(result.asset_class.weights) >= 4
    assert len(result.market_cap.weights) >= 3


def test_concentrated_portfolio():
    """All in one category → low score, findings generated."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="IT Stock 1", asset_type=AssetType.EQUITY, sector="IT", market_value=500000),
        _make_holding(name="IT Stock 2", asset_type=AssetType.EQUITY, sector="IT", market_value=300000),
        _make_holding(name="IT Stock 3", asset_type=AssetType.EQUITY, sector="IT", market_value=200000),
    ])
    result = analyze_diversification(portfolio)
    assert result.score < 30
    # Should have findings about equity concentration and no debt
    assert len(result.findings) > 0
    # Should have equity sector breakdown
    assert result.equity_sector is not None
    assert "IT" in result.equity_sector.weights


def test_findings_high_equity():
    """90%+ equity should generate high severity finding."""
    portfolio = Portfolio(holdings=[
        _make_holding(sector="Large Cap Fund", market_value=900000),
        _make_holding(sector="Liquid Fund", market_value=100000),
    ])
    result = analyze_diversification(portfolio)
    # 90% equity should trigger high severity
    assert any(f.severity == "high" and "90%" in f.message for f in result.findings)


def test_no_debt_finding():
    """100% equity should note absence of debt."""
    portfolio = Portfolio(holdings=[
        _make_holding(sector="Large Cap Fund", market_value=500000),
        _make_holding(sector="Mid Cap Fund", market_value=500000),
    ])
    result = analyze_diversification(portfolio)
    debt_findings = [f for f in result.findings if "debt" in f.message.lower()]
    assert len(debt_findings) > 0


def test_holdings_without_value():
    """Holdings with no market_value and no invested_value should be skipped."""
    portfolio = Portfolio(holdings=[
        _make_holding(market_value=None, invested_value=None),
        _make_holding(market_value=100000),
    ])
    result = analyze_diversification(portfolio)
    assert result.holdings_analyzed == 2
    assert result.holdings_with_value == 1


def test_invested_value_fallback():
    """Should use invested_value when market_value is missing."""
    portfolio = Portfolio(holdings=[
        _make_holding(market_value=None, invested_value=50000, sector="Large Cap Fund"),
        _make_holding(market_value=50000, invested_value=40000, sector="Liquid Fund"),
    ])
    result = analyze_diversification(portfolio)
    assert result.holdings_with_value == 2
    assert abs(result.asset_class.buckets[AssetClass.EQUITY] - 50000) < 0.01
    assert abs(result.asset_class.buckets[AssetClass.DEBT] - 50000) < 0.01


def test_real_portfolio_integration():
    """Test with a realistic Indian portfolio mix."""
    portfolio = Portfolio(holdings=[
        # MFs
        _make_holding(name="SBI Bluechip Fund", sector="Large Cap Fund", market_value=200000),
        _make_holding(name="Axis Midcap Fund", sector="Mid Cap Fund", market_value=150000),
        _make_holding(name="Tata Digital India Fund", sector="Sectoral/Thematic", market_value=100000),
        _make_holding(name="SBI Small Cap Fund", sector="Small Cap Fund", market_value=80000),
        _make_holding(name="Parag Parikh Flexi Cap", sector="Flexi Cap Fund", market_value=200000),
        # ETFs
        _make_holding(name="Nippon Gold BeES", sector="Gold ETF", asset_type=AssetType.ETF, market_value=50000),
        _make_holding(name="Nifty BeES", sector="Other  ETFs", asset_type=AssetType.ETF, market_value=100000),
        # Stocks
        _make_holding(name="TCS", asset_type=AssetType.EQUITY, sector="IT", market_value=80000),
        _make_holding(name="HDFC Bank", asset_type=AssetType.EQUITY, sector="Banking", market_value=60000),
        _make_holding(name="Reliance", asset_type=AssetType.EQUITY, sector="Oil & Gas", market_value=50000),
    ])
    result = analyze_diversification(portfolio)

    # Should have reasonable score (mixed portfolio)
    assert 30 < result.score < 80

    # Asset class should have equity + gold
    assert AssetClass.EQUITY in result.asset_class.weights
    assert AssetClass.GOLD in result.asset_class.weights

    # Market cap should cover large, mid, small, multi_flexi
    assert MarketCap.LARGE in result.market_cap.weights
    assert MarketCap.MID in result.market_cap.weights
    assert MarketCap.SMALL in result.market_cap.weights

    # Equity sector breakdown should exist
    assert result.equity_sector is not None
    assert "IT" in result.equity_sector.weights
    assert "Banking" in result.equity_sector.weights


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
        except Exception:  # noqa: BLE001 — script runner reports every failure
            print(f"  FAIL  {test_fn.__name__}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
