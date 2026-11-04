"""Risk Analyzer — Phase 4 Lens (5a in roadmap).

Analyzes portfolio concentration risk at the individual holding level:
1. Single-holding concentration (any holding > 10% of portfolio)
2. Top-N concentration (top 3/5/10 holdings as % of portfolio)
3. Volatility bucket estimation from asset composition
4. Overall risk level classification

Concentrated vs diversified instruments (D-013): a holding is *concentrated*
when it is direct equity, a Sectoral/Thematic fund, or an MF/ETF with no
recognisable AMFI category; every other MF/ETF category is *diversified*.
Single-stock style concentration penalties (single-holding schedule,
count-over-10 %, top-3 share) apply to concentrated instruments only; a
diversified fund carries a smaller single-manager penalty instead. The
risk-bucket penalty separates the level-5 share (mid/small cap, sectoral,
direct equity) from the all-high share (levels 4+5, i.e. broad equity funds
included), so an index fund is not scored like a small-cap fund.

All computation is deterministic. No LLM calls. No historical price data needed.
"""

from dataclasses import dataclass, field

from lenses.models import AssetType, Holding, Portfolio

# ---------------------------------------------------------------------------
# Risk classification from AMFI categories
# ---------------------------------------------------------------------------

# Higher number = higher risk. Scale: 1-5
_CATEGORY_RISK: dict[str, int] = {
    # Low risk (1)
    "Overnight Fund": 1,
    "Liquid Fund": 1,
    "Money Market Fund": 1,
    # Low-moderate (2)
    "Ultra Short Duration Fund": 2,
    "Low Duration Fund": 2,
    "Short Duration Fund": 2,
    "Floater Fund": 2,
    "Banking and PSU Fund": 2,
    "Arbitrage Fund": 2,
    # Moderate (3)
    "Medium Duration Fund": 3,
    "Medium to Long Duration Fund": 3,
    "Corporate Bond Fund": 3,
    "Dynamic Bond": 3,
    "Gilt Fund": 3,
    "Gilt Fund with 10 year constant duration": 3,
    "Credit Risk Fund": 3,
    "Long Duration Fund": 3,
    "Conservative Hybrid Fund": 3,
    "Equity Savings": 3,
    "Gold ETF": 3,
    "Gold Fund": 3,
    # Moderate-high (4)
    "Balanced Advantage Fund": 4,
    "Aggressive Hybrid Fund": 4,
    "Multi Asset Allocation": 4,
    "Large Cap Fund": 4,
    "Index Funds": 4,
    "Other  ETFs": 4,  # double space in AMFI data
    "Fund of Funds (Overseas)": 4,
    "Fund of Funds (Domestic)": 4,
    "Large & Mid Cap Fund": 4,
    "Multi Cap Fund": 4,
    "Flexi Cap Fund": 4,
    "Dividend Yield Fund": 4,
    "Value Fund": 4,
    "Contra Fund": 4,
    "Focused Fund": 4,
    "ELSS": 4,
    # High risk (5)
    "Mid Cap Fund": 5,
    "Small Cap Fund": 5,
    "Sectoral/Thematic": 5,
}


def _holding_risk_level(h: Holding) -> int:
    """Estimate risk level (1-5) for a single holding."""
    # Direct equity is always high risk
    if h.asset_type == AssetType.EQUITY:
        return 5

    cat = h.sector or ""
    if cat in _CATEGORY_RISK:
        return _CATEGORY_RISK[cat]

    # Fallback by asset type
    if h.asset_type == AssetType.ETF:
        return 4
    if h.asset_type == AssetType.MUTUAL_FUND:
        return 4  # default for unknown MF categories

    return 3


def _is_concentrated(h: Holding) -> bool:
    """True for instruments whose concentration is single-security risk (D-013).

    Direct equity, Sectoral/Thematic funds, and MF/ETF holdings without a
    recognisable AMFI category are concentrated. Every other MF/ETF category
    holds a diversified basket and is treated as a single-manager exposure.
    """
    if h.asset_type == AssetType.EQUITY:
        return True
    cat = h.sector or ""
    if cat == "Sectoral/Thematic":
        return True
    return cat not in _CATEGORY_RISK


@dataclass
class _RiskInputs:
    """Shares (% of portfolio) the score and label are computed from."""
    max_concentrated_pct: float  # largest concentrated instrument
    max_diversified_pct: float  # largest diversified fund
    concentrated_over_10: int  # concentrated instruments at >= 10 %
    concentrated_top_3_pct: float  # top-3 concentrated instruments' combined share
    concentrated_pct: float  # all concentrated instruments' combined share
    level_5_pct: float  # risk level 5 share
    all_high_pct: float  # risk level 4 + 5 share
    low_pct: float  # risk level 1-2 share


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class HoldingConcentration:
    """A holding that represents significant portfolio concentration."""
    scheme_name: str
    isin: str | None
    asset_type: str
    market_value: float
    weight_pct: float  # % of total portfolio
    risk_level: int  # 1-5


@dataclass
class TopNConcentration:
    """Concentration metrics for top-N holdings."""
    top_3_pct: float
    top_5_pct: float
    top_10_pct: float
    total_holdings: int


@dataclass
class RiskBuckets:
    """Portfolio breakdown by risk level."""
    low: float  # % in risk level 1-2
    moderate: float  # % in risk level 3
    high: float  # % in risk level 4-5
    weights: dict[str, float]  # "low" | "moderate" | "high" → weight (0-1)


@dataclass
class RiskResult:
    """Full output of the Risk Analyzer lens."""
    score: float  # 0-100, higher = lower risk (better balanced)
    risk_label: str  # "low" | "moderate" | "high" | "very_high"

    # Concentration analysis
    concentrated_holdings: list[HoldingConcentration] = field(default_factory=list)
    top_n: TopNConcentration | None = None

    # Risk composition
    risk_buckets: RiskBuckets | None = None

    findings: list[str] = field(default_factory=list)
    holdings_analyzed: int = 0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_risk(portfolio: Portfolio) -> RiskResult:
    """Run the risk analysis on a portfolio.

    Phase 1 lens — deterministic, no user input needed.
    """
    holdings = portfolio.holdings
    if not holdings:
        return RiskResult(
            score=50.0,
            risk_label="moderate",
            holdings_analyzed=0,
        )

    def _value(h: Holding) -> float:
        return h.market_value or h.invested_value or 0.0

    valued = [(h, _value(h)) for h in holdings if _value(h) > 0]
    total_value = sum(v for _, v in valued)

    if total_value <= 0:
        return RiskResult(
            score=50.0,
            risk_label="moderate",
            holdings_analyzed=len(holdings),
        )

    # --- 1. Single-holding concentration ---
    holding_weights = [
        (h, val, val / total_value * 100)
        for h, val in valued
    ]
    holding_weights.sort(key=lambda x: x[2], reverse=True)

    concentrated = [
        HoldingConcentration(
            scheme_name=h.scheme_name,
            isin=h.isin,
            asset_type=h.asset_type.value,
            market_value=val,
            weight_pct=round(pct, 1),
            risk_level=_holding_risk_level(h),
        )
        for h, val, pct in holding_weights
        if pct >= 5  # Include holdings >= 5% for display
    ]

    # --- 2. Top-N concentration ---
    sorted_values = sorted([v for _, v in valued], reverse=True)
    top_3_pct = sum(sorted_values[:3]) / total_value * 100 if len(sorted_values) >= 3 else 100.0
    top_5_pct = sum(sorted_values[:5]) / total_value * 100 if len(sorted_values) >= 5 else 100.0
    top_10_pct = sum(sorted_values[:10]) / total_value * 100 if len(sorted_values) >= 10 else 100.0

    top_n = TopNConcentration(
        top_3_pct=round(top_3_pct, 1),
        top_5_pct=round(top_5_pct, 1),
        top_10_pct=round(top_10_pct, 1),
        total_holdings=len(valued),
    )

    # --- 3. Risk bucket breakdown ---
    low_value = 0.0
    moderate_value = 0.0
    high_value = 0.0

    for h, val in valued:
        rl = _holding_risk_level(h)
        if rl <= 2:
            low_value += val
        elif rl == 3:
            moderate_value += val
        else:
            high_value += val

    risk_buckets = RiskBuckets(
        low=round(low_value / total_value * 100, 1),
        moderate=round(moderate_value / total_value * 100, 1),
        high=round(high_value / total_value * 100, 1),
        weights={
            "low": low_value / total_value if total_value > 0 else 0,
            "moderate": moderate_value / total_value if total_value > 0 else 0,
            "high": high_value / total_value if total_value > 0 else 0,
        },
    )

    # --- 4. Scoring ---
    conc_pcts = sorted(
        (pct for h, _, pct in holding_weights if _is_concentrated(h)), reverse=True,
    )
    div_pcts = [pct for h, _, pct in holding_weights if not _is_concentrated(h)]
    level_5_value = sum(val for h, val in valued if _holding_risk_level(h) == 5)
    inputs = _RiskInputs(
        max_concentrated_pct=round(conc_pcts[0], 1) if conc_pcts else 0.0,
        max_diversified_pct=round(max(div_pcts), 1) if div_pcts else 0.0,
        concentrated_over_10=sum(1 for pct in conc_pcts if round(pct, 1) >= 10),
        concentrated_top_3_pct=round(sum(conc_pcts[:3]), 1),
        concentrated_pct=round(sum(conc_pcts), 1),
        level_5_pct=round(level_5_value / total_value * 100, 1),
        all_high_pct=risk_buckets.high,
        low_pct=risk_buckets.low,
    )
    score, risk_label = _compute_risk_score(inputs, len(valued))

    # --- 5. Findings ---
    findings = _generate_findings(
        concentrated, top_n, risk_buckets, len(valued),
    )

    return RiskResult(
        score=score,
        risk_label=risk_label,
        concentrated_holdings=concentrated,
        top_n=top_n,
        risk_buckets=risk_buckets,
        findings=findings,
        holdings_analyzed=len(holdings),
    )


def _compute_risk_score(inputs: _RiskInputs, num_holdings: int) -> tuple[float, str]:
    """Compute risk score (0-100, higher = better risk profile).

    Penalizes for (D-013):
    - Single-holding concentration: concentrated instruments 35/25/15/5 at
      >= 30/20/10/5 %; diversified funds (single manager) 20/10/5 at >= 50/30/20 %
    - Concentrated instruments at >= 10 % (5 each, max 15)
    - Top-3 concentrated instruments' share 20/12/5 at >= 80/60/40 %
    - Level-5 share 25/15/8 at >= 60/40/25 %; all-high (4+5) share 15/10/5 at
      >= 90/80/70 %; no low-risk allocation 10
    - Few holdings 10/5 at < 3/< 5, only when concentrated instruments are at
      least half the portfolio
    """
    penalty = 0.0

    # Single-holding concentration — concentrated instruments (0-35 points)
    max_conc = inputs.max_concentrated_pct
    if max_conc >= 30:
        penalty += 35
    elif max_conc >= 20:
        penalty += 25
    elif max_conc >= 10:
        penalty += 15
    elif max_conc >= 5:
        penalty += 5

    # Single-manager concentration — diversified funds (0-20 points)
    max_div = inputs.max_diversified_pct
    if max_div >= 50:
        penalty += 20
    elif max_div >= 30:
        penalty += 10
    elif max_div >= 20:
        penalty += 5

    # Count of concentrated instruments at >= 10 % (0-15 points)
    penalty += min(inputs.concentrated_over_10 * 5, 15)

    # Top-3 concentrated instruments' share of the portfolio (0-20 points)
    top_3 = inputs.concentrated_top_3_pct
    if top_3 >= 80:
        penalty += 20
    elif top_3 >= 60:
        penalty += 12
    elif top_3 >= 40:
        penalty += 5

    # Level-5 share: mid/small cap, sectoral, direct equity (0-25 points)
    level_5 = inputs.level_5_pct
    if level_5 >= 60:
        penalty += 25
    elif level_5 >= 40:
        penalty += 15
    elif level_5 >= 25:
        penalty += 8

    # All-high share: risk levels 4 + 5 (0-15 points)
    all_high = inputs.all_high_pct
    if all_high >= 90:
        penalty += 15
    elif all_high >= 80:
        penalty += 10
    elif all_high >= 70:
        penalty += 5

    # No low-risk allocation (10 points)
    if inputs.low_pct == 0:
        penalty += 10

    # Few holdings (0-10 points), only for a mostly-concentrated portfolio
    if inputs.concentrated_pct >= 50:
        if num_holdings < 3:
            penalty += 10
        elif num_holdings < 5:
            penalty += 5

    score = round(max(100 - penalty, 0), 1)

    # Risk label — derived from actual portfolio composition, not the score.
    # Score measures how well-diversified your risk is (concentration),
    # but the label should reflect the actual risk level of your holdings.
    if level_5 >= 50 or (all_high >= 90 and level_5 >= 25):
        label = "very_high"
    elif all_high >= 60:
        label = "high"
    elif inputs.low_pct >= 60:
        label = "low"
    else:
        label = "moderate"

    return score, label


def _generate_findings(
    concentrated: list[HoldingConcentration],
    top_n: TopNConcentration,
    risk_buckets: RiskBuckets,
    num_holdings: int,
) -> list[str]:
    """Generate plain-English findings about portfolio risk."""
    findings: list[str] = []

    # Single-holding concentration
    over_10 = [c for c in concentrated if c.weight_pct >= 10]
    if over_10:
        names = ", ".join(f"'{c.scheme_name}' ({c.weight_pct:.0f}%)" for c in over_10[:3])
        findings.append(
            f"{len(over_10)} holding{'s' if len(over_10) > 1 else ''} "
            f"{'each represent' if len(over_10) > 1 else 'represents'} "
            f"over 10% of your portfolio: {names}. "
            f"This is a high level of single-holding concentration."
        )

    # Top-3 concentration
    if top_n and top_n.top_3_pct >= 60 and num_holdings > 5:
        findings.append(
            f"Your top 3 holdings account for {top_n.top_3_pct:.0f}% of "
            f"your portfolio value — a high level of concentration."
        )

    # Risk bucket imbalance
    if risk_buckets:
        if risk_buckets.high >= 90:
            findings.append(
                f"{risk_buckets.high:.0f}% of your portfolio is in high-risk "
                f"assets (equity, mid/small cap MFs, sectoral funds) — "
                f"a very high concentration in volatile asset classes."
            )
        elif risk_buckets.high >= 75:
            findings.append(
                f"{risk_buckets.high:.0f}% of your portfolio is in higher-risk "
                f"assets, with no debt or hybrid fund exposure."
            )

        if risk_buckets.low == 0 and num_holdings > 3:
            findings.append(
                "Your portfolio has no low-risk allocation (liquid funds, "
                "short-duration debt, overnight funds)."
            )

    # Too few holdings
    if num_holdings < 5:
        findings.append(
            f"Your portfolio has only {num_holdings} holdings. "
            f"For reference, diversified portfolios commonly have 10-20 holdings "
            f"across different asset classes."
        )

    return findings
