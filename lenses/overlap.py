"""Overlap Detector — Phase 1 Lens (3b in roadmap, Lens 2 in product spec).

Detects common underlying stocks across MFs, ETFs, and direct equity holdings.
Computes pairwise overlap using Jaccard similarity on stock-weight vectors.
Flags when two or more funds share >30% of their holdings by weight.

Overlap types detected:
1. MF ↔ MF — Two funds holding the same stocks
2. MF/ETF ↔ Direct Equity — User holds a stock directly AND through a fund
3. ETF ↔ MF — Index ETF overlapping with an active fund
4. ETF ↔ ETF — Two ETFs tracking similar indices

All computation is deterministic. No LLM calls.

TODO: Direct equity overlap matching is ISIN-exact only. Add fuzzy name matching
      (e.g., "TCS" vs "Tata Consultancy Services Ltd") to catch overlaps where
      ISIN is not available for the underlying holding.
TODO: Morningstar API requests have a 1.5s delay between calls, making the upload
      endpoint slow for portfolios with many funds (e.g., 8 funds ≈ 12+ seconds).
      Consider async/parallel fetching or background processing.
"""

import logging
from dataclasses import dataclass, field

from lenses.models import SchemePortfolio
from lenses.models import AssetType, Holding, Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def _build_weight_vector(portfolio: SchemePortfolio) -> dict[str, float]:
    """Build a normalized weight vector from a scheme's holdings.

    Key is the security name (normalized uppercase). Value is weight (0-100).
    We use security name as the key since ISINs may not always be available
    for underlying holdings.
    """
    vec: dict[str, float] = {}
    for h in portfolio.holdings:
        # Use ISIN when available (reliable cross-fund matching).
        # Fall back to normalized name (less reliable but better than nothing).
        # Never use None/empty as key — that collapses unrelated holdings.
        isin = h.isin if (h.isin and h.isin.lower() not in ("none", "nan", "")) else None
        key = isin or h.security_name.upper().strip()
        if not key:
            continue
        # Some holdings appear multiple times (e.g., different share classes)
        vec[key] = vec.get(key, 0) + h.weight
    return vec


def _jaccard_weighted(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute weighted Jaccard similarity between two weight vectors.

    For each shared stock, takes the minimum weight from either fund.
    Result is the sum of minimums divided by the sum of maximums.
    Returns value between 0.0 (no overlap) and 1.0 (identical).
    """
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    if not all_keys:
        return 0.0

    min_sum = 0.0
    max_sum = 0.0
    for key in all_keys:
        w_a = vec_a.get(key, 0)
        w_b = vec_b.get(key, 0)
        min_sum += min(w_a, w_b)
        max_sum += max(w_a, w_b)

    if max_sum == 0:
        return 0.0
    return min_sum / max_sum


def _stock_overlap_count(vec_a: dict[str, float], vec_b: dict[str, float]) -> int:
    """Count the number of stocks that appear in both vectors."""
    return len(set(vec_a.keys()) & set(vec_b.keys()))


def _shared_stocks(
    vec_a: dict[str, float], vec_b: dict[str, float],
    portfolio_map: dict[str, SchemePortfolio],
    isin_a: str, isin_b: str,
) -> list[dict]:
    """Get the list of shared stocks with their weights in each fund."""
    shared_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not shared_keys:
        return []

    # Build a reverse map from key to display name
    name_map: dict[str, str] = {}
    for isin in (isin_a, isin_b):
        sp = portfolio_map.get(isin)
        if sp:
            for h in sp.holdings:
                key = h.isin or h.security_name.upper().strip()
                if key not in name_map:
                    name_map[key] = h.security_name

    result = []
    for key in sorted(shared_keys, key=lambda k: min(vec_a[k], vec_b[k]), reverse=True):
        result.append({
            "stock": name_map.get(key, key),
            "weight_a": round(vec_a[key], 2),
            "weight_b": round(vec_b[key], 2),
        })
    return result


# ---------------------------------------------------------------------------
# Direct equity overlap detection
# ---------------------------------------------------------------------------

def _find_direct_equity_overlaps(
    portfolio: Portfolio,
    scheme_portfolios: dict[str, SchemePortfolio],
) -> list["EquityOverlapFinding"]:
    """Find stocks held directly that also appear in MF/ETF holdings."""
    equity_holdings = [
        h for h in portfolio.holdings
        if h.asset_type == AssetType.EQUITY and (h.isin or h.symbol)
    ]
    if not equity_holdings:
        return []

    # Build lookup: stock ISIN/name → direct holding
    direct_stocks: dict[str, Holding] = {}
    for h in equity_holdings:
        if h.isin:
            direct_stocks[h.isin] = h
        name_key = h.scheme_name.upper().strip()
        direct_stocks[name_key] = h

    findings: list[EquityOverlapFinding] = []

    for scheme_isin, sp in scheme_portfolios.items():
        for uh in sp.holdings:
            stock_key = uh.isin or uh.security_name.upper().strip()
            matched_holding = direct_stocks.get(stock_key)

            if not matched_holding:
                # Try matching by partial name (e.g., "TCS" vs "Tata Consultancy Services")
                # Skip for now — exact match only to avoid false positives
                continue

            findings.append(EquityOverlapFinding(
                stock_name=matched_holding.scheme_name,
                stock_isin=matched_holding.isin,
                fund_name=sp.scheme_name,
                fund_isin=sp.scheme_isin,
                weight_in_fund=uh.weight,
                direct_value=matched_holding.market_value or matched_holding.invested_value or 0,
            ))

    return findings


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class FundPairOverlap:
    """Overlap between two funds (MF/ETF)."""
    fund_a_name: str
    fund_a_isin: str
    fund_b_name: str
    fund_b_isin: str
    overlap_pct: float  # Jaccard similarity as percentage (0-100)
    shared_stock_count: int
    total_stocks_a: int
    total_stocks_b: int
    top_shared_stocks: list[dict] = field(default_factory=list)
    severity: str = "info"  # "high" | "moderate" | "info"


@dataclass
class EquityOverlapFinding:
    """A direct stock that also appears in a MF/ETF."""
    stock_name: str
    stock_isin: str | None
    fund_name: str
    fund_isin: str
    weight_in_fund: float  # weight % of this stock in the fund
    direct_value: float  # market value of direct holding


@dataclass
class OverlapResult:
    """Full output of the Overlap Detector lens."""
    score: float  # 0-100, higher = less overlap (better)

    fund_pairs: list[FundPairOverlap] = field(default_factory=list)
    equity_overlaps: list[EquityOverlapFinding] = field(default_factory=list)

    findings: list[str] = field(default_factory=list)  # plain-English findings

    schemes_analyzed: int = 0  # MFs/ETFs with available holdings data
    schemes_failed: int = 0  # MFs/ETFs where holdings fetch failed
    total_fund_pairs: int = 0  # total pairwise comparisons made


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

_OVERLAP_HIGH_THRESHOLD = 40  # % — Jaccard similarity above this is high severity
_OVERLAP_MODERATE_THRESHOLD = 25  # % — above this is moderate
_OVERLAP_FLAG_THRESHOLD = 15  # % — below this we don't report


def analyze_overlap(
    portfolio: Portfolio,
    scheme_portfolios: dict[str, SchemePortfolio],
) -> OverlapResult:
    """Run the overlap analysis on a portfolio.

    Args:
        portfolio: The user's merged portfolio.
        scheme_portfolios: Dict mapping ISIN → SchemePortfolio with underlying
            holdings data (fetched from Morningstar or other source).

    Returns:
        OverlapResult with pairwise overlaps, equity overlaps, and score.
    """
    fund_holdings = [
        h for h in portfolio.holdings
        if h.asset_type in (AssetType.MUTUAL_FUND, AssetType.ETF) and h.isin
    ]

    # Build weight vectors for each scheme with available data
    weight_vectors: dict[str, dict[str, float]] = {}
    schemes_with_data: list[Holding] = []

    for h in fund_holdings:
        sp = scheme_portfolios.get(h.isin)
        if sp and sp.holdings:
            weight_vectors[h.isin] = _build_weight_vector(sp)
            schemes_with_data.append(h)

    schemes_failed = len(fund_holdings) - len(schemes_with_data)

    # Pairwise overlap computation
    fund_pairs: list[FundPairOverlap] = []
    n = len(schemes_with_data)

    for i in range(n):
        for j in range(i + 1, n):
            h_a = schemes_with_data[i]
            h_b = schemes_with_data[j]
            vec_a = weight_vectors[h_a.isin]
            vec_b = weight_vectors[h_b.isin]

            overlap = _jaccard_weighted(vec_a, vec_b)
            if overlap != overlap:  # NaN check
                continue
            overlap_pct = round(overlap * 100, 1)

            if overlap_pct < _OVERLAP_FLAG_THRESHOLD:
                continue

            shared_count = _stock_overlap_count(vec_a, vec_b)
            top_shared = _shared_stocks(
                vec_a, vec_b, scheme_portfolios, h_a.isin, h_b.isin,
            )

            severity = "info"
            if overlap_pct >= _OVERLAP_HIGH_THRESHOLD:
                severity = "high"
            elif overlap_pct >= _OVERLAP_MODERATE_THRESHOLD:
                severity = "moderate"

            sp_a = scheme_portfolios[h_a.isin]
            sp_b = scheme_portfolios[h_b.isin]

            fund_pairs.append(FundPairOverlap(
                fund_a_name=h_a.scheme_name,
                fund_a_isin=h_a.isin,
                fund_b_name=h_b.scheme_name,
                fund_b_isin=h_b.isin,
                overlap_pct=overlap_pct,
                shared_stock_count=shared_count,
                total_stocks_a=len(sp_a.holdings),
                total_stocks_b=len(sp_b.holdings),
                top_shared_stocks=top_shared[:10],  # top 10 shared stocks
                severity=severity,
            ))

    # Sort by overlap percentage descending
    fund_pairs.sort(key=lambda p: p.overlap_pct, reverse=True)

    # Direct equity overlaps
    equity_overlaps = _find_direct_equity_overlaps(portfolio, scheme_portfolios)

    # Generate findings
    findings = _generate_findings(fund_pairs, equity_overlaps)

    # Compute score
    score = _compute_overlap_score(fund_pairs, equity_overlaps, n)

    total_pairs = n * (n - 1) // 2

    return OverlapResult(
        score=score,
        fund_pairs=fund_pairs,
        equity_overlaps=equity_overlaps,
        findings=findings,
        schemes_analyzed=len(schemes_with_data),
        schemes_failed=schemes_failed,
        total_fund_pairs=total_pairs,
    )


def _compute_overlap_score(
    fund_pairs: list[FundPairOverlap],
    equity_overlaps: list[EquityOverlapFinding],
    num_schemes: int,
) -> float:
    """Compute overlap sub-score (0-100, higher = less overlap = better).

    Fund-to-fund overlap is the primary driver. Direct equity overlaps are
    a secondary signal — each individual overlap is minor (user holds a stock
    that happens to be in a broad fund), so we penalize based on the total
    weight of overlap, not the count.
    """
    if num_schemes <= 1 and not equity_overlaps:
        return 100.0  # Nothing to overlap with

    # Fund pair penalties (primary signal)
    pair_penalty = 0.0
    for pair in fund_pairs:
        if pair.overlap_pct >= _OVERLAP_HIGH_THRESHOLD:
            pair_penalty += 25.0
        elif pair.overlap_pct >= _OVERLAP_MODERATE_THRESHOLD:
            pair_penalty += 15.0
        else:
            pair_penalty += 5.0

    # Direct equity overlap penalties — weight-based, not count-based
    # Only penalize overlaps with significant fund weight (>2%)
    significant_eq_overlaps = [e for e in equity_overlaps if e.weight_in_fund >= 2.0]
    equity_penalty = min(len(significant_eq_overlaps) * 2.0, 30.0)

    total_penalty = min(pair_penalty + equity_penalty, 100.0)
    return round(max(100.0 - total_penalty, 0.0), 1)


def _generate_findings(
    fund_pairs: list[FundPairOverlap],
    equity_overlaps: list[EquityOverlapFinding],
) -> list[str]:
    """Generate plain-English findings about portfolio overlap."""
    findings: list[str] = []

    # High-overlap fund pairs
    high_pairs = [p for p in fund_pairs if p.severity == "high"]
    moderate_pairs = [p for p in fund_pairs if p.severity == "moderate"]

    for pair in high_pairs:
        findings.append(
            f"'{pair.fund_a_name}' and '{pair.fund_b_name}' have "
            f"{pair.overlap_pct:.0f}% portfolio overlap — {pair.shared_stock_count} "
            f"stocks in common. You may be paying two expense ratios for "
            f"largely the same exposure."
        )

    for pair in moderate_pairs:
        findings.append(
            f"'{pair.fund_a_name}' and '{pair.fund_b_name}' share "
            f"{pair.shared_stock_count} stocks ({pair.overlap_pct:.0f}% overlap)."
        )

    # Direct equity overlaps
    if equity_overlaps:
        # Group by stock
        by_stock: dict[str, list[EquityOverlapFinding]] = {}
        for eo in equity_overlaps:
            by_stock.setdefault(eo.stock_name, []).append(eo)

        for stock, overlaps in by_stock.items():
            fund_names = [o.fund_name for o in overlaps]
            if len(fund_names) == 1:
                findings.append(
                    f"You hold {stock} directly and also through "
                    f"'{fund_names[0]}' (where it is {overlaps[0].weight_in_fund:.1f}% "
                    f"of the fund)."
                )
            else:
                names_str = ", ".join(f"'{n}'" for n in fund_names[:3])
                findings.append(
                    f"You hold {stock} directly and also through "
                    f"{len(fund_names)} funds: {names_str}."
                )

    if not findings:
        findings.append(
            "No significant overlap detected between your fund holdings."
        )

    return findings
