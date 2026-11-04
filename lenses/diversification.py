"""Diversification Analyzer — Phase 1 Lens (3a).

Analyzes portfolio diversification across three dimensions:
1. Asset class split (equity / debt / hybrid / gold / international / cash)
   — HHI, anchored to three classes (equity, debt, one diversifier) (D-014)
2. Market cap distribution (large / mid / small / multi-flexi) — distance from
   the broad-market cap mix (D-014, see below)
3. Sector/category concentration — HHI over the categories held

Market cap (D-014): equity funds are bucketed by AMFI category; index funds and
"Other ETFs" are bucketed from the index named in the scheme name, and ELSS is
treated as multi/flexi cap. The buckets decompose into effective large/mid/small
exposure (large & mid → 50/50/0, multi/flexi → 70/20/10) and the sub-score is
100 × (1 − L1(w, market) / 2) with market = 70/20/10 (approximate Nifty 500).
100 % large cap scores 70, 100 % flexi cap 100, 100 % small cap 10. Unclassified
exposure is left out of that vector.

All computation is deterministic. No LLM calls.
Uses AMFI scheme categories as the classification source for MFs/ETFs.
Uses broker-provided sector data for direct equities.

TODO: Use underlying MF holdings (from Morningstar/AMC data) for true sector
      concentration analysis instead of AMFI category as proxy. Currently,
      "Sectoral/Thematic" funds are treated as a single category, not broken
      down by actual sector exposure.
TODO: Market cap classification is "unclassified" for Sectoral/Thematic, Value,
      Contra, Dividend Yield, and Focused funds, and for direct equity. Use
      underlying holdings data to determine their actual large/mid/small split.
"""

from dataclasses import dataclass, field

from lenses.models import AssetType, Holding, Portfolio

# ---------------------------------------------------------------------------
# AMFI category → asset class mapping
# ---------------------------------------------------------------------------

_EQUITY_CATEGORIES = {
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund",
    "Large & Mid Cap Fund", "Multi Cap Fund", "Flexi Cap Fund",
    "ELSS", "Sectoral/Thematic", "Value Fund", "Contra Fund",
    "Dividend Yield Fund", "Focused Fund", "Index Funds",
    "Other  ETFs",  # note: double space in AMFI data
}

_DEBT_CATEGORIES = {
    "Liquid Fund", "Ultra Short Duration Fund", "Low Duration Fund",
    "Short Duration Fund", "Medium Duration Fund", "Medium to Long Duration Fund",
    "Long Duration Fund", "Dynamic Bond", "Corporate Bond Fund",
    "Credit Risk Fund", "Banking and PSU Fund", "Gilt Fund",
    "Gilt Fund with 10 year constant duration", "Money Market Fund",
    "Overnight Fund", "Floater Fund", "Index Funds/ETFs",
}

_HYBRID_CATEGORIES = {
    "Balanced Advantage Fund", "Aggressive Hybrid Fund",
    "Conservative Hybrid Fund", "Multi Asset Allocation",
    "Arbitrage Fund", "Equity Savings",
}

_GOLD_CATEGORIES = {
    "Gold ETF", "Gold Fund",
}

_INTERNATIONAL_CATEGORIES = {
    "Fund of Funds (Overseas)",
}


class AssetClass:
    EQUITY = "equity"
    DEBT = "debt"
    HYBRID = "hybrid"
    GOLD = "gold"
    INTERNATIONAL = "international"
    OTHER = "other"


def _classify_asset_class(holding: Holding) -> str:
    """Map a holding to a broad asset class using its sector (AMFI category)."""
    if holding.asset_type == AssetType.EQUITY:
        return AssetClass.EQUITY

    cat = holding.sector or ""

    if cat in _GOLD_CATEGORIES:
        return AssetClass.GOLD
    if cat in _INTERNATIONAL_CATEGORIES:
        return AssetClass.INTERNATIONAL
    if cat in _DEBT_CATEGORIES:
        return AssetClass.DEBT
    if cat in _HYBRID_CATEGORIES:
        return AssetClass.HYBRID
    if cat in _EQUITY_CATEGORIES:
        return AssetClass.EQUITY

    # Fallback: check for keywords when AMFI category isn't set
    name_upper = (holding.scheme_name or "").upper()
    if any(kw in name_upper for kw in ("LIQUID", "DEBT", "BOND", "GILT", "MONEY MARKET")):
        return AssetClass.DEBT
    if any(kw in name_upper for kw in ("GOLD", "SILVER")):
        return AssetClass.GOLD
    if "BALANCED" in name_upper or "HYBRID" in name_upper:
        return AssetClass.HYBRID

    # MFs/ETFs without category default to equity (most common)
    if holding.asset_type in (AssetType.MUTUAL_FUND, AssetType.ETF):
        return AssetClass.EQUITY

    return AssetClass.OTHER


# ---------------------------------------------------------------------------
# AMFI category → market cap mapping (equity MFs only)
# ---------------------------------------------------------------------------

class MarketCap:
    LARGE = "large_cap"
    MID = "mid_cap"
    SMALL = "small_cap"
    LARGE_MID = "large_mid_cap"
    MULTI_FLEXI = "multi_flexi_cap"
    UNCLASSIFIED = "unclassified"


_MARKET_CAP_MAP: dict[str, str] = {
    "Large Cap Fund": MarketCap.LARGE,
    "Mid Cap Fund": MarketCap.MID,
    "Small Cap Fund": MarketCap.SMALL,
    "Large & Mid Cap Fund": MarketCap.LARGE_MID,
    "Multi Cap Fund": MarketCap.MULTI_FLEXI,
    "Flexi Cap Fund": MarketCap.MULTI_FLEXI,
}


# Categories whose cap mix is read from the index named in the scheme name.
_INDEX_CATEGORIES = {"Index Funds", "Other  ETFs"}  # note: double space in AMFI data

# Checked in order: small and mid before the broad/large tokens, and
# "NIFTY 500" before "NIFTY 50" (a substring of it).
_INDEX_NAME_CAP: list[tuple[tuple[str, ...], str]] = [
    (("SMALLCAP", "SMALL CAP"), MarketCap.SMALL),
    (("MIDCAP", "MID CAP"), MarketCap.MID),
    (("NIFTY 500", "TOTAL MARKET", "BSE 500"), MarketCap.MULTI_FLEXI),
    (("NIFTY 50", "SENSEX", "NIFTY 100", "BSE 100", "NIFTY NEXT 50", "JUNIOR"),
     MarketCap.LARGE),
]


def _classify_index_by_name(scheme_name: str) -> str:
    """Market cap of an index fund / ETF from the index in its name."""
    name = " ".join((scheme_name or "").upper().split())
    for tokens, cap in _INDEX_NAME_CAP:
        if any(t in name for t in tokens):
            return cap
    return MarketCap.UNCLASSIFIED


def _classify_market_cap(holding: Holding) -> str | None:
    """Classify market cap for equity MFs. Returns None for non-equity or unclassifiable."""
    asset_class = _classify_asset_class(holding)
    if asset_class != AssetClass.EQUITY:
        return None

    cat = holding.sector or ""
    if cat in _INDEX_CATEGORIES:
        return _classify_index_by_name(holding.scheme_name)
    if cat == "ELSS":
        return MarketCap.MULTI_FLEXI  # no cap mandate; run as a flexi-cap strategy
    return _MARKET_CAP_MAP.get(cat, MarketCap.UNCLASSIFIED)


# Effective (large, mid, small) exposure of each market-cap bucket.
_CAP_DECOMPOSITION: dict[str, tuple[float, float, float]] = {
    MarketCap.LARGE: (1.0, 0.0, 0.0),
    MarketCap.MID: (0.0, 1.0, 0.0),
    MarketCap.SMALL: (0.0, 0.0, 1.0),
    MarketCap.LARGE_MID: (0.5, 0.5, 0.0),
    MarketCap.MULTI_FLEXI: (0.7, 0.2, 0.1),
}

# Approximate Nifty 500 large / mid / small split.
_MARKET_CAP_MIX: tuple[float, float, float] = (0.70, 0.20, 0.10)


def _market_cap_score(cap_buckets: dict[str, float]) -> float:
    """Score 0-100 as closeness of effective large/mid/small exposure to the market.

    100 × (1 − L1(w, market) / 2). Unclassified exposure is excluded; with no
    classified exposure the score is 0.
    """
    effective = [0.0, 0.0, 0.0]
    for bucket, value in cap_buckets.items():
        split = _CAP_DECOMPOSITION.get(bucket)
        if split is None:
            continue
        for i, share in enumerate(split):
            effective[i] += value * share

    total = sum(effective)
    if total <= 0:
        return 0.0

    l1 = sum(abs(e / total - m) for e, m in zip(effective, _MARKET_CAP_MIX))
    return round(100 * (1 - l1 / 2), 1)


# ---------------------------------------------------------------------------
# HHI (Herfindahl-Hirschman Index)
# ---------------------------------------------------------------------------

def _compute_hhi(weights: dict[str, float]) -> float:
    """Compute HHI from weight dict. Weights should sum to ~1.0.

    Returns value between 0 and 10000.
    - 10000 = maximally concentrated (single holding)
    - < 1500 = well diversified
    - 1500-2500 = moderately concentrated
    - > 2500 = highly concentrated
    """
    return sum((w * 100) ** 2 for w in weights.values())


def _weights_from_values(buckets: dict[str, float]) -> dict[str, float]:
    """Convert absolute values to proportional weights (summing to 1.0)."""
    total = sum(buckets.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in buckets.items() if v > 0}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _hhi_to_score(hhi: float, max_categories: int) -> float:
    """Convert HHI to a 0-100 score where higher = more diversified.

    Uses the theoretical minimum HHI (perfect equal distribution) as anchor.
    If max_categories=1, returns 0 (can't be diversified with 1 bucket).
    """
    if max_categories <= 1:
        return 0.0

    # Theoretical HHI for perfect equal distribution
    min_hhi = 10000 / max_categories
    max_hhi = 10000  # Single bucket

    if hhi <= min_hhi:
        return 100.0
    if hhi >= max_hhi:
        return 0.0

    # Linear interpolation
    return round((max_hhi - hhi) / (max_hhi - min_hhi) * 100, 1)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

@dataclass
class AllocationBreakdown:
    """Value breakdown by bucket (e.g., asset class or market cap)."""
    buckets: dict[str, float] = field(default_factory=dict)  # bucket → market value
    weights: dict[str, float] = field(default_factory=dict)  # bucket → weight (0-1)
    hhi: float = 0.0
    score: float = 0.0  # 0-100 sub-score


@dataclass
class ConcentrationFinding:
    """A specific finding about portfolio concentration."""
    category: str  # e.g., "IT" or "Large Cap Fund"
    weight_pct: float  # e.g., 45.2
    severity: str  # "high" | "moderate" | "info"
    message: str


@dataclass
class DiversificationResult:
    """Full output of the Diversification Analyzer lens."""
    score: float  # Composite 0-100

    asset_class: AllocationBreakdown
    market_cap: AllocationBreakdown
    category_concentration: AllocationBreakdown  # MF/ETF category-level

    findings: list[ConcentrationFinding] = field(default_factory=list)
    holdings_analyzed: int = 0
    holdings_with_value: int = 0  # Holdings that have market_value

    # Equity-specific breakdown (for direct equity holdings)
    equity_sector: AllocationBreakdown | None = None


def analyze_diversification(portfolio: Portfolio) -> DiversificationResult:
    """Run the diversification analysis on a portfolio.

    This is a Phase 1 lens — runs immediately on upload, no user input needed.
    All computation is deterministic.
    """
    holdings = portfolio.holdings
    if not holdings:
        return DiversificationResult(
            score=0.0,
            asset_class=AllocationBreakdown(),
            market_cap=AllocationBreakdown(),
            category_concentration=AllocationBreakdown(),
            holdings_analyzed=0,
            holdings_with_value=0,
        )

    # Use market_value where available, fall back to invested_value
    def _value(h: Holding) -> float:
        return h.market_value or h.invested_value or 0.0

    valued_holdings = [h for h in holdings if _value(h) > 0]

    # --- 1. Asset class breakdown ---
    asset_buckets: dict[str, float] = {}
    for h in valued_holdings:
        ac = _classify_asset_class(h)
        asset_buckets[ac] = asset_buckets.get(ac, 0) + _value(h)

    asset_weights = _weights_from_values(asset_buckets)
    asset_hhi = _compute_hhi(asset_weights)
    # Anchor: equity, debt and one diversifier = 3 (D-014)
    asset_score = _hhi_to_score(asset_hhi, max_categories=3)

    # --- 2. Market cap distribution (equity MFs only) ---
    cap_buckets: dict[str, float] = {}
    equity_mf_total = 0.0
    for h in valued_holdings:
        cap = _classify_market_cap(h)
        if cap is not None:
            cap_buckets[cap] = cap_buckets.get(cap, 0) + _value(h)
            equity_mf_total += _value(h)

    cap_weights = _weights_from_values(cap_buckets)
    cap_hhi = _compute_hhi(cap_weights)  # informational; not the score (D-014)
    cap_score = _market_cap_score(cap_buckets)

    # --- 3. Category/sector concentration ---
    # For MFs/ETFs: use AMFI category. For equities: use broker sector.
    cat_buckets: dict[str, float] = {}
    for h in valued_holdings:
        cat = h.sector or "Uncategorized"
        cat_buckets[cat] = cat_buckets.get(cat, 0) + _value(h)

    cat_weights = _weights_from_values(cat_buckets)
    cat_hhi = _compute_hhi(cat_weights)
    # Many possible categories — use actual count
    cat_score = _hhi_to_score(cat_hhi, max_categories=max(len(cat_buckets), 2))

    # --- 4. Equity sector breakdown (direct stocks only) ---
    equity_sector: AllocationBreakdown | None = None
    equity_holdings = [h for h in valued_holdings if h.asset_type == AssetType.EQUITY]
    if equity_holdings:
        eq_buckets: dict[str, float] = {}
        for h in equity_holdings:
            sec = h.sector or "Uncategorized"
            eq_buckets[sec] = eq_buckets.get(sec, 0) + _value(h)
        eq_weights = _weights_from_values(eq_buckets)
        eq_hhi = _compute_hhi(eq_weights)
        eq_score = _hhi_to_score(eq_hhi, max_categories=max(len(eq_buckets), 2))
        equity_sector = AllocationBreakdown(
            buckets=eq_buckets, weights=eq_weights, hhi=eq_hhi, score=eq_score,
        )

    # --- 5. Generate findings ---
    findings = _generate_findings(
        asset_weights, cap_weights, cat_weights, equity_sector,
    )

    # --- 6. Composite score ---
    # Weighted: asset class 40%, market cap 30%, category 30%
    # If no market cap data (no equity MFs), redistribute: asset 50%, category 50%
    if cap_buckets:
        composite = 0.40 * asset_score + 0.30 * cap_score + 0.30 * cat_score
    else:
        composite = 0.50 * asset_score + 0.50 * cat_score

    return DiversificationResult(
        score=round(composite, 1),
        asset_class=AllocationBreakdown(
            buckets=asset_buckets, weights=asset_weights,
            hhi=asset_hhi, score=asset_score,
        ),
        market_cap=AllocationBreakdown(
            buckets=cap_buckets, weights=cap_weights,
            hhi=cap_hhi, score=cap_score,
        ),
        category_concentration=AllocationBreakdown(
            buckets=cat_buckets, weights=cat_weights,
            hhi=cat_hhi, score=cat_score,
        ),
        findings=findings,
        holdings_analyzed=len(holdings),
        holdings_with_value=len(valued_holdings),
        equity_sector=equity_sector,
    )


def _generate_findings(
    asset_weights: dict[str, float],
    cap_weights: dict[str, float],
    cat_weights: dict[str, float],
    equity_sector: AllocationBreakdown | None,
) -> list[ConcentrationFinding]:
    """Generate plain-English findings from the analysis."""
    findings: list[ConcentrationFinding] = []

    # Asset class concentration
    equity_pct = asset_weights.get(AssetClass.EQUITY, 0) * 100
    debt_pct = asset_weights.get(AssetClass.DEBT, 0) * 100

    if equity_pct >= 90:
        findings.append(ConcentrationFinding(
            category="Asset Class",
            weight_pct=round(equity_pct, 1),
            severity="high",
            message=(
                f"Your portfolio is {equity_pct:.0f}% equity with minimal "
                f"debt or other asset classes — a very high single-asset-class "
                f"concentration."
            ),
        ))
    elif equity_pct >= 75:
        findings.append(ConcentrationFinding(
            category="Asset Class",
            weight_pct=round(equity_pct, 1),
            severity="moderate",
            message=(
                f"Your portfolio is {equity_pct:.0f}% equity — a high concentration "
                f"in a single asset class."
            ),
        ))

    if debt_pct == 0 and equity_pct > 0:
        findings.append(ConcentrationFinding(
            category="Asset Class",
            weight_pct=0.0,
            severity="moderate",
            message=(
                "Your portfolio has no debt allocation. For reference, "
                "diversified portfolios commonly include a debt component."
            ),
        ))

    gold_pct = asset_weights.get(AssetClass.GOLD, 0) * 100
    intl_pct = asset_weights.get(AssetClass.INTERNATIONAL, 0) * 100

    if gold_pct == 0 and intl_pct == 0 and len(asset_weights) > 0:
        findings.append(ConcentrationFinding(
            category="Asset Class",
            weight_pct=0.0,
            severity="info",
            message=(
                "No allocation to gold or international assets. These asset "
                "classes have historically shown low correlation with Indian equities."
            ),
        ))

    # Market cap concentration
    if cap_weights:
        large_pct = cap_weights.get(MarketCap.LARGE, 0) * 100
        small_pct = cap_weights.get(MarketCap.SMALL, 0) * 100

        if large_pct >= 80:
            findings.append(ConcentrationFinding(
                category="Market Cap",
                weight_pct=round(large_pct, 1),
                severity="moderate",
                message=(
                    f"Equity allocation is {large_pct:.0f}% large-cap with "
                    f"limited mid-cap and small-cap exposure."
                ),
            ))

        if small_pct >= 40:
            findings.append(ConcentrationFinding(
                category="Market Cap",
                weight_pct=round(small_pct, 1),
                severity="high",
                message=(
                    f"Small-cap allocation is {small_pct:.0f}% of equity. "
                    f"Small caps are among the most volatile market-cap segments."
                ),
            ))

        unclassified_pct = cap_weights.get(MarketCap.UNCLASSIFIED, 0) * 100
        if unclassified_pct >= 50:
            findings.append(ConcentrationFinding(
                category="Market Cap",
                weight_pct=round(unclassified_pct, 1),
                severity="info",
                message=(
                    f"{unclassified_pct:.0f}% of equity MF allocation is in "
                    f"thematic/sectoral/ELSS funds where market-cap distribution "
                    f"cannot be determined from fund category alone."
                ),
            ))

    # Category concentration (top-heavy)
    if cat_weights:
        sorted_cats = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_weight = sorted_cats[0]
        top_pct = top_weight * 100

        if top_pct >= 50 and len(sorted_cats) > 1:
            findings.append(ConcentrationFinding(
                category="Category Concentration",
                weight_pct=round(top_pct, 1),
                severity="high" if top_pct >= 70 else "moderate",
                message=(
                    f"'{top_cat}' accounts for {top_pct:.0f}% of your portfolio — "
                    f"a high concentration in a single category."
                ),
            ))

    # Equity sector concentration (direct stocks)
    if equity_sector and equity_sector.weights:
        sorted_sectors = sorted(
            equity_sector.weights.items(), key=lambda x: x[1], reverse=True,
        )
        top_sec, top_sec_weight = sorted_sectors[0]
        top_sec_pct = top_sec_weight * 100

        if top_sec_pct >= 40 and len(sorted_sectors) > 1:
            findings.append(ConcentrationFinding(
                category="Equity Sector",
                weight_pct=round(top_sec_pct, 1),
                severity="high" if top_sec_pct >= 60 else "moderate",
                message=(
                    f"Direct equity holdings are {top_sec_pct:.0f}% concentrated "
                    f"in {top_sec} — a high single-sector concentration."
                ),
            ))

    return findings
