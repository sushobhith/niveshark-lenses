"""Tests for the rupee-cost aggregator."""

import pytest

from lenses.rupee_cost import compute_rupee_cost, _compute_overlap_drag, _get_expense_ratio
from lenses.cost import CostResult, CostSummary
from lenses.overlap import OverlapResult, FundPairOverlap
from lenses.tax import TaxResult, TaxSummary
from lenses.models import AssetType, Holding, Portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_holding(
    name: str,
    isin: str | None = None,
    asset_type: AssetType = AssetType.MUTUAL_FUND,
    market_value: float = 100_000,
    sector: str | None = "Large Cap Fund",
) -> Holding:
    return Holding(
        scheme_name=name,
        isin=isin,
        asset_type=asset_type,
        units=100,
        market_value=market_value,
        invested_value=market_value,
        sector=sector,
        source="test",
    )


def _make_portfolio(holdings: list[Holding]) -> Portfolio:
    return Portfolio(holdings=holdings, sources_used=["test"])


def _make_cost_result(savings: float = 5000) -> CostResult:
    return CostResult(
        score=60,
        summary=CostSummary(
            total_annual_cost=10000,
            total_potential_savings=savings,
            regular_plan_count=3,
            direct_plan_count=2,
            unclassified_count=0,
            avg_expense_ratio_pct=1.2,
        ),
    )


def _make_tax_result(total_tax: float = 8000) -> TaxResult:
    return TaxResult(
        score=70,
        summary=TaxSummary(
            total_unrealized_gain=50000,
            total_unrealized_loss=5000,
            net_unrealized=45000,
            estimated_equity_tax=6000,
            estimated_debt_tax=2000,
            estimated_total_tax=total_tax,
            harvest_potential=5000,
            elss_count=1,
        ),
    )


def _make_overlap_result(pairs: list[FundPairOverlap] | None = None) -> OverlapResult:
    return OverlapResult(
        score=50,
        fund_pairs=pairs or [],
        schemes_analyzed=4,
    )


# ---------------------------------------------------------------------------
# Tests: compute_rupee_cost (aggregator)
# ---------------------------------------------------------------------------

class TestComputeRupeeCost:
    def test_all_lenses_available(self):
        portfolio = _make_portfolio([
            _make_holding("Fund A", isin="INF001", market_value=200_000),
            _make_holding("Fund B", isin="INF002", market_value=100_000),
        ])
        cost = _make_cost_result(savings=4890)
        tax = _make_tax_result(total_tax=8181)
        overlap = _make_overlap_result()

        result = compute_rupee_cost(cost, overlap, tax, portfolio)

        assert result.cost_drag == 4890
        assert result.tax_liability == 8181
        assert result.total_annual_drag == 4890  # no overlap pairs above 25%
        assert result.disclaimer

    def test_no_cost_lens(self):
        portfolio = _make_portfolio([])
        result = compute_rupee_cost(None, None, _make_tax_result(3000), portfolio)

        assert result.cost_drag == 0
        assert result.overlap_drag == 0
        assert result.tax_liability == 3000
        assert result.total_annual_drag == 0

    def test_no_lenses_at_all(self):
        portfolio = _make_portfolio([])
        result = compute_rupee_cost(None, None, None, portfolio)

        assert result.total_annual_drag == 0
        assert result.cost_drag == 0
        assert result.overlap_drag == 0
        assert result.tax_liability == 0

    def test_with_overlap_drag(self):
        portfolio = _make_portfolio([
            _make_holding("Fund A Regular", isin="INF001", market_value=200_000, sector="Large Cap Fund"),
            _make_holding("Fund B Regular", isin="INF002", market_value=100_000, sector="Mid Cap Fund"),
        ])
        overlap = _make_overlap_result(pairs=[
            FundPairOverlap(
                fund_a_name="Fund A", fund_a_isin="INF001",
                fund_b_name="Fund B", fund_b_isin="INF002",
                overlap_pct=40, shared_stock_count=10,
                total_stocks_a=30, total_stocks_b=25,
                severity="high",
            ),
        ])
        result = compute_rupee_cost(
            _make_cost_result(0), overlap, _make_tax_result(0), portfolio,
        )

        # Smaller fund = 100K, overlap = 40%, avg ER = (1.60 + 1.80) / 2 = 1.70%
        # Drag = 100000 * 0.40 * 0.0170 = 680
        assert result.overlap_drag == pytest.approx(680, abs=1)
        assert result.total_annual_drag == pytest.approx(680, abs=1)


# ---------------------------------------------------------------------------
# Tests: overlap drag calculation
# ---------------------------------------------------------------------------

class TestOverlapDrag:
    def test_no_pairs(self):
        portfolio = _make_portfolio([])
        overlap = _make_overlap_result()
        assert _compute_overlap_drag(overlap, portfolio) == 0

    def test_pair_below_threshold(self):
        """Pairs with < 25% overlap are skipped."""
        portfolio = _make_portfolio([
            _make_holding("A", isin="INF001", market_value=100_000),
            _make_holding("B", isin="INF002", market_value=100_000),
        ])
        overlap = _make_overlap_result(pairs=[
            FundPairOverlap(
                fund_a_name="A", fund_a_isin="INF001",
                fund_b_name="B", fund_b_isin="INF002",
                overlap_pct=20, shared_stock_count=5,
                total_stocks_a=20, total_stocks_b=20,
            ),
        ])
        assert _compute_overlap_drag(overlap, portfolio) == 0

    def test_missing_holdings_skipped(self):
        """Pairs where holdings can't be found are skipped."""
        portfolio = _make_portfolio([
            _make_holding("A", isin="INF001", market_value=100_000),
        ])
        overlap = _make_overlap_result(pairs=[
            FundPairOverlap(
                fund_a_name="A", fund_a_isin="INF001",
                fund_b_name="B", fund_b_isin="INF999",  # not in portfolio
                overlap_pct=50, shared_stock_count=10,
                total_stocks_a=20, total_stocks_b=20,
            ),
        ])
        assert _compute_overlap_drag(overlap, portfolio) == 0

    def test_none_overlap_result(self):
        portfolio = _make_portfolio([])
        assert _compute_overlap_drag(None, portfolio) == 0


# ---------------------------------------------------------------------------
# Tests: expense ratio lookup
# ---------------------------------------------------------------------------

class TestGetExpenseRatio:
    def test_known_category(self):
        h = _make_holding("Test Regular", sector="Large Cap Fund")
        assert _get_expense_ratio(h) == 1.60

    def test_direct_plan_from_name(self):
        h = _make_holding("HDFC Large Cap Direct Growth", sector="Large Cap Fund")
        assert _get_expense_ratio(h) == 0.55

    def test_equity_stock(self):
        h = _make_holding("TCS", asset_type=AssetType.EQUITY, sector=None)
        assert _get_expense_ratio(h) == 0.0

    def test_etf_fallback(self):
        h = _make_holding("Nifty ETF", asset_type=AssetType.ETF, sector="Unknown")
        assert _get_expense_ratio(h) == 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
