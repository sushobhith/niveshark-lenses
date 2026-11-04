"""Tests for the Overlap Detector lens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses.overlap import (
    OverlapResult,
    _build_weight_vector,
    _jaccard_weighted,
    _stock_overlap_count,
    _compute_overlap_score,
    analyze_overlap,
)
from lenses.models import SchemePortfolio, UnderlyingHolding
from lenses.models import AssetType, Holding, Portfolio


def _make_holding(
    name: str = "Test Fund",
    isin: str | None = "INF000000001",
    asset_type: AssetType = AssetType.MUTUAL_FUND,
    market_value: float | None = 100000.0,
    units: float = 100.0,
    source: str = "test",
) -> Holding:
    return Holding(
        scheme_name=name, isin=isin, asset_type=asset_type,
        market_value=market_value, units=units, source=source,
    )


def _make_scheme_portfolio(
    isin: str, name: str, holdings: list[tuple[str, float, str | None]],
) -> SchemePortfolio:
    """Create a SchemePortfolio. Holdings: list of (stock_name, weight, isin)."""
    return SchemePortfolio(
        scheme_isin=isin,
        scheme_name=name,
        provider_id="test",
        holdings=[
            UnderlyingHolding(
                security_name=h[0],
                weight=h[1],
                isin=h[2] if len(h) > 2 else None,
            )
            for h in holdings
        ],
        total_holdings_count=len(holdings),
    )


# ---------------------------------------------------------------------------
# Unit tests: Jaccard similarity
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    """Identical vectors → similarity 1.0."""
    vec = {"A": 50, "B": 30, "C": 20}
    assert _jaccard_weighted(vec, vec) == 1.0


def test_jaccard_no_overlap():
    """No common stocks → similarity 0.0."""
    vec_a = {"A": 50, "B": 50}
    vec_b = {"C": 50, "D": 50}
    assert _jaccard_weighted(vec_a, vec_b) == 0.0


def test_jaccard_partial_overlap():
    """Partial overlap."""
    vec_a = {"A": 50, "B": 50}
    vec_b = {"A": 50, "C": 50}
    # min_sum = min(50,50) + min(50,0) + min(0,50) = 50
    # max_sum = max(50,50) + max(50,0) + max(0,50) = 50 + 50 + 50 = 150
    result = _jaccard_weighted(vec_a, vec_b)
    assert abs(result - 50 / 150) < 0.001


def test_jaccard_empty():
    assert _jaccard_weighted({}, {}) == 0.0
    assert _jaccard_weighted({"A": 50}, {}) == 0.0


def test_stock_overlap_count():
    vec_a = {"A": 50, "B": 30, "C": 20}
    vec_b = {"A": 40, "C": 60}
    assert _stock_overlap_count(vec_a, vec_b) == 2


# ---------------------------------------------------------------------------
# Unit tests: weight vector building
# ---------------------------------------------------------------------------

def test_build_weight_vector():
    sp = _make_scheme_portfolio("INF001", "Fund A", [
        ("HDFC Bank", 10.0, "INE040A01034"),
        ("Reliance", 8.0, "INE002A01018"),
        ("TCS", 5.0, None),
    ])
    vec = _build_weight_vector(sp)
    assert vec["INE040A01034"] == 10.0
    assert vec["INE002A01018"] == 8.0
    assert vec["TCS"] == 5.0  # Falls back to name when no ISIN


def test_build_weight_vector_deduplicates():
    """Same ISIN appearing twice should have weights summed."""
    sp = _make_scheme_portfolio("INF001", "Fund A", [
        ("HDFC Bank", 5.0, "INE040A01034"),
        ("HDFC Bank Ltd", 3.0, "INE040A01034"),
    ])
    vec = _build_weight_vector(sp)
    assert vec["INE040A01034"] == 8.0


# ---------------------------------------------------------------------------
# Integration tests: full overlap analysis
# ---------------------------------------------------------------------------

def test_no_funds():
    """Portfolio with only equities → no overlap analysis."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="TCS", asset_type=AssetType.EQUITY),
    ])
    result = analyze_overlap(portfolio, {})
    assert result.score == 100.0
    assert result.schemes_analyzed == 0


def test_single_fund():
    """Single fund → nothing to compare, score 100."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
    ])
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 10.0, "INE040A01034"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert result.score == 100.0
    assert result.schemes_analyzed == 1


def test_high_overlap_funds():
    """Two funds with 100% identical holdings → high overlap flagged."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
    ])
    identical_holdings = [
        ("HDFC Bank", 30.0, "INE040A01034"),
        ("Reliance", 25.0, "INE002A01018"),
        ("TCS", 20.0, "INE467B01029"),
        ("Infosys", 15.0, "INE009A01021"),
        ("ICICI Bank", 10.0, "INE090A01021"),
    ]
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", identical_holdings),
        "INF002": _make_scheme_portfolio("INF002", "Fund B", identical_holdings),
    }
    result = analyze_overlap(portfolio, sp)
    assert result.score < 80  # Penalized
    assert len(result.fund_pairs) == 1
    assert result.fund_pairs[0].overlap_pct == 100.0
    assert result.fund_pairs[0].severity == "high"
    assert len(result.findings) > 0


def test_no_overlap_funds():
    """Two completely different funds → no overlap flagged."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
    ])
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 50.0, "INE040A01034"),
            ("Reliance", 50.0, "INE002A01018"),
        ]),
        "INF002": _make_scheme_portfolio("INF002", "Fund B", [
            ("Wipro", 50.0, "INE075A01022"),
            ("Bharti Airtel", 50.0, "INE397D01024"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert result.score == 100.0
    assert len(result.fund_pairs) == 0


def test_moderate_overlap():
    """Funds with some shared stocks → moderate overlap."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
    ])
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 30.0, "INE040A01034"),
            ("Reliance", 30.0, "INE002A01018"),
            ("TCS", 20.0, "INE467B01029"),
            ("Infosys", 20.0, "INE009A01021"),
        ]),
        "INF002": _make_scheme_portfolio("INF002", "Fund B", [
            ("HDFC Bank", 25.0, "INE040A01034"),
            ("Reliance", 25.0, "INE002A01018"),
            ("Wipro", 25.0, "INE075A01022"),
            ("Bharti Airtel", 25.0, "INE397D01024"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert len(result.fund_pairs) >= 1
    pair = result.fund_pairs[0]
    assert 15 < pair.overlap_pct < 70
    assert pair.shared_stock_count == 2


def test_direct_equity_overlap():
    """Stock held directly and also in a fund."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(
            name="HDFC Bank", isin="INE040A01034",
            asset_type=AssetType.EQUITY, market_value=50000,
        ),
    ])
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 10.0, "INE040A01034"),
            ("Reliance", 15.0, "INE002A01018"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert len(result.equity_overlaps) == 1
    eo = result.equity_overlaps[0]
    assert eo.stock_name == "HDFC Bank"
    assert eo.fund_name == "Fund A"
    assert eo.weight_in_fund == 10.0


def test_missing_holdings_data():
    """Funds without Morningstar data should be counted as failed."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
    ])
    # Only Fund A has data
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 50.0, "INE040A01034"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert result.schemes_analyzed == 1
    assert result.schemes_failed == 1


def test_three_funds_pairwise():
    """Three funds → 3 pairwise comparisons."""
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
        _make_holding(name="Fund C", isin="INF003"),
    ])
    # All share some stocks
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", [
            ("HDFC Bank", 40.0, "INE040A01034"),
            ("Reliance", 30.0, "INE002A01018"),
            ("TCS", 30.0, "INE467B01029"),
        ]),
        "INF002": _make_scheme_portfolio("INF002", "Fund B", [
            ("HDFC Bank", 35.0, "INE040A01034"),
            ("Infosys", 35.0, "INE009A01021"),
            ("TCS", 30.0, "INE467B01029"),
        ]),
        "INF003": _make_scheme_portfolio("INF003", "Fund C", [
            ("Wipro", 50.0, "INE075A01022"),
            ("Bharti Airtel", 50.0, "INE397D01024"),
        ]),
    }
    result = analyze_overlap(portfolio, sp)
    assert result.total_fund_pairs == 3
    assert result.schemes_analyzed == 3
    # A-B should have overlap, A-C and B-C should not
    overlapping = [p for p in result.fund_pairs if p.overlap_pct > 0]
    assert len(overlapping) >= 1


def test_top_shared_stocks_limited():
    """Top shared stocks should be limited to 10."""
    # Create two funds sharing 15 stocks
    shared = [(f"Stock_{i}", 100 / 15, f"INE{i:03d}") for i in range(15)]
    portfolio = Portfolio(holdings=[
        _make_holding(name="Fund A", isin="INF001"),
        _make_holding(name="Fund B", isin="INF002"),
    ])
    sp = {
        "INF001": _make_scheme_portfolio("INF001", "Fund A", shared),
        "INF002": _make_scheme_portfolio("INF002", "Fund B", shared),
    }
    result = analyze_overlap(portfolio, sp)
    assert len(result.fund_pairs) == 1
    assert len(result.fund_pairs[0].top_shared_stocks) <= 10


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
