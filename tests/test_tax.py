"""Tests for the Tax Awareness Analyzer lens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.tax import (
    TaxResult,
    TaxRegime,
    analyze_tax,
    _classify_tax_regime,
    _is_elss,
)
from lenses.models import AssetType, Holding, Portfolio


def _make_holding(
    name: str = "Test Fund",
    isin: str | None = "INF000000001",
    asset_type: AssetType = AssetType.MUTUAL_FUND,
    sector: str | None = None,
    market_value: float | None = 100000.0,
    invested_value: float | None = 80000.0,
    units: float = 100.0,
    source: str = "test",
) -> Holding:
    return Holding(
        scheme_name=name, isin=isin, asset_type=asset_type, sector=sector,
        market_value=market_value, invested_value=invested_value,
        units=units, source=source,
    )


def _make_portfolio(holdings: list[Holding]) -> Portfolio:
    return Portfolio(holdings=holdings, sources_used=["test"])


# ---------------------------------------------------------------------------
# Unit tests: tax regime classification
# ---------------------------------------------------------------------------

def test_regime_direct_equity():
    h = _make_holding(asset_type=AssetType.EQUITY)
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_etf():
    h = _make_holding(asset_type=AssetType.ETF, sector="Other  ETFs")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_large_cap():
    h = _make_holding(sector="Large Cap Fund")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_small_cap():
    h = _make_holding(sector="Small Cap Fund")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_liquid_fund():
    h = _make_holding(sector="Liquid Fund")
    assert _classify_tax_regime(h) == TaxRegime.DEBT


def test_regime_gilt_fund():
    h = _make_holding(sector="Gilt Fund")
    assert _classify_tax_regime(h) == TaxRegime.DEBT


def test_regime_overnight():
    h = _make_holding(sector="Overnight Fund")
    assert _classify_tax_regime(h) == TaxRegime.DEBT


def test_regime_aggressive_hybrid():
    h = _make_holding(sector="Aggressive Hybrid Fund")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_conservative_hybrid():
    h = _make_holding(sector="Conservative Hybrid Fund")
    assert _classify_tax_regime(h) == TaxRegime.DEBT


def test_regime_multi_asset():
    h = _make_holding(sector="Multi Asset Allocation")
    assert _classify_tax_regime(h) == TaxRegime.HYBRID_EQUITY


def test_regime_gold_etf():
    h = _make_holding(sector="Gold ETF", asset_type=AssetType.ETF)
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_elss():
    h = _make_holding(sector="ELSS")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_unknown_defaults_equity():
    h = _make_holding(sector="Some Unknown Category")
    assert _classify_tax_regime(h) == TaxRegime.EQUITY


def test_regime_fallback_debt_keyword():
    h = _make_holding(name="HDFC Liquid Fund", sector=None)
    assert _classify_tax_regime(h) == TaxRegime.DEBT


# ---------------------------------------------------------------------------
# Unit tests: ELSS detection
# ---------------------------------------------------------------------------

def test_elss_by_category():
    h = _make_holding(sector="ELSS")
    assert _is_elss(h) is True


def test_elss_by_name():
    h = _make_holding(name="Mirae Asset ELSS Tax Saver Fund")
    assert _is_elss(h) is True


def test_elss_tax_saver():
    h = _make_holding(name="Axis Tax Saver Fund")
    assert _is_elss(h) is True


def test_not_elss():
    h = _make_holding(name="SBI Large Cap Fund", sector="Large Cap Fund")
    assert _is_elss(h) is False


# ---------------------------------------------------------------------------
# Integration tests: analyze_tax
# ---------------------------------------------------------------------------

def test_empty_portfolio():
    result = analyze_tax(_make_portfolio([]))
    assert result.score == 50.0
    assert result.holdings_analyzed == 0
    assert result.summary.estimated_total_tax == 0


def test_portfolio_with_gains():
    """Portfolio with unrealized gains should have tax estimate."""
    portfolio = _make_portfolio([
        _make_holding("SBI Large Cap", sector="Large Cap Fund",
                       market_value=150000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    assert result.summary.total_unrealized_gain == 50000
    assert result.summary.estimated_total_tax > 0
    assert len(result.regime_breakdown) > 0
    assert result.limitation_note != ""


def test_portfolio_with_losses():
    """Portfolio at loss should show harvest candidates."""
    portfolio = _make_portfolio([
        _make_holding("SBI Small Cap", sector="Small Cap Fund",
                       market_value=70000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    assert result.summary.total_unrealized_loss == 30000
    assert result.summary.net_unrealized == -30000
    assert result.summary.estimated_total_tax == 0
    assert len(result.harvest_candidates) == 1
    assert result.harvest_candidates[0].unrealized_loss == -30000


def test_mixed_gains_and_losses():
    """Mix of gains and losses across regimes."""
    portfolio = _make_portfolio([
        _make_holding("Large Cap", sector="Large Cap Fund",
                       market_value=200000, invested_value=100000),  # +100k equity gain
        _make_holding("Liquid Fund", sector="Liquid Fund",
                       market_value=95000, invested_value=100000),  # -5k debt loss
    ])
    result = analyze_tax(portfolio)
    assert result.summary.total_unrealized_gain == 100000
    assert result.summary.total_unrealized_loss == 5000
    assert len(result.harvest_candidates) == 1
    assert result.harvest_candidates[0].scheme_name == "Liquid Fund"


def test_regime_breakdown_counts():
    """Regime breakdown should separate equity and debt."""
    portfolio = _make_portfolio([
        _make_holding("Equity Fund", sector="Large Cap Fund",
                       market_value=200000, invested_value=150000),
        _make_holding("Debt Fund", sector="Liquid Fund",
                       market_value=100000, invested_value=90000),
    ])
    result = analyze_tax(portfolio)
    regimes = {b.regime for b in result.regime_breakdown}
    assert TaxRegime.EQUITY in regimes
    assert TaxRegime.DEBT in regimes


def test_elss_detection():
    """ELSS holdings should be flagged."""
    portfolio = _make_portfolio([
        _make_holding("Mirae ELSS Tax Saver", sector="ELSS",
                       market_value=100000, invested_value=80000),
        _make_holding("SBI Large Cap", sector="Large Cap Fund",
                       market_value=100000, invested_value=90000),
    ])
    result = analyze_tax(portfolio)
    assert len(result.elss_holdings) == 1
    assert result.elss_holdings[0].scheme_name == "Mirae ELSS Tax Saver"
    assert result.summary.elss_count == 1


def test_tax_rates_equity():
    """Equity regime should have correct rates."""
    portfolio = _make_portfolio([
        _make_holding("Large Cap", sector="Large Cap Fund",
                       market_value=200000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    equity_breakdown = [b for b in result.regime_breakdown if b.regime == TaxRegime.EQUITY]
    assert len(equity_breakdown) == 1
    assert equity_breakdown[0].ltcg_rate_pct == 12.5
    assert equity_breakdown[0].stcg_rate_pct == 20.0


def test_tax_rates_debt():
    """Debt regime should have slab rate."""
    portfolio = _make_portfolio([
        _make_holding("Liquid", sector="Liquid Fund",
                       market_value=110000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    debt_breakdown = [b for b in result.regime_breakdown if b.regime == TaxRegime.DEBT]
    assert len(debt_breakdown) == 1
    assert debt_breakdown[0].ltcg_rate_pct == 30.0
    assert debt_breakdown[0].stcg_rate_pct == 30.0


def test_worst_case_equity_tax():
    """Equity STCG at 20% for worst case."""
    portfolio = _make_portfolio([
        _make_holding("Equity", sector="Large Cap Fund",
                       market_value=200000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    # 100k gain * 20% STCG = 20000
    assert result.summary.estimated_equity_tax == 20000


def test_worst_case_debt_tax():
    """Debt at 30% slab rate for worst case."""
    portfolio = _make_portfolio([
        _make_holding("Debt", sector="Liquid Fund",
                       market_value=200000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    # 100k gain * 30% = 30000
    assert result.summary.estimated_debt_tax == 30000


def test_no_value_holdings_skipped():
    """Holdings with no value should be skipped."""
    portfolio = _make_portfolio([
        _make_holding("No Value", market_value=None, invested_value=None),
        _make_holding("Has Value", market_value=100000, invested_value=80000),
    ])
    result = analyze_tax(portfolio)
    assert result.summary.total_unrealized_gain == 20000


def test_score_net_loss_portfolio():
    """Portfolio at net loss should have high score (no tax liability)."""
    portfolio = _make_portfolio([
        _make_holding("Loser", market_value=50000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    assert result.score >= 80


def test_harvest_sorted_by_loss():
    """Harvest candidates should be sorted by largest loss first."""
    portfolio = _make_portfolio([
        _make_holding("Small Loss", market_value=95000, invested_value=100000),
        _make_holding("Big Loss", market_value=60000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    assert len(result.harvest_candidates) == 2
    assert result.harvest_candidates[0].scheme_name == "Big Loss"


def test_findings_generated():
    """Should produce findings for portfolio with gains."""
    portfolio = _make_portfolio([
        _make_holding("Fund", sector="Large Cap Fund",
                       market_value=200000, invested_value=100000),
    ])
    result = analyze_tax(portfolio)
    assert len(result.findings) > 0


def test_limitation_note_always_present():
    """Limitation note should always be present."""
    result = analyze_tax(_make_portfolio([
        _make_holding("Fund", market_value=100000, invested_value=80000),
    ]))
    assert "transaction dates" in result.limitation_note.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
