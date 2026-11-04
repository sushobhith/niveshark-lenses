"""Tax Awareness Analyzer — Phase 4 Lens (5b in roadmap).

Provides tax-regime awareness for portfolio holdings. Since CAS statements
and broker exports don't include purchase dates, we cannot classify holdings
as LTCG or STCG. Instead, this lens:

1. Classifies each holding into its tax regime (equity / debt / hybrid)
2. Breaks down unrealized gains/losses by tax regime
3. Identifies tax-loss harvesting candidates
4. Flags ELSS holdings (3-year lock-in)
5. Estimates worst-case tax liability if liquidated today

Indian tax rules (FY 2024-25, applicable from July 2024):
- Equity (MF equity, ETF, stocks):
    LTCG (>12 months): 12.5% above ₹1.25L exemption
    STCG (≤12 months): 20%
- Debt MF (purchased after Apr 2023): Taxed at slab rate (no indexation)
- Hybrid depends on equity allocation (≥65% equity → equity taxation)

All computation is deterministic. No LLM calls.
"""

from dataclasses import dataclass, field

from lenses.models import AssetType, Holding, Portfolio


# ---------------------------------------------------------------------------
# Tax regime classification using AMFI categories
# ---------------------------------------------------------------------------

class TaxRegime:
    EQUITY = "equity"  # 12.5% LTCG / 20% STCG
    DEBT = "debt"  # Slab rate (post Apr 2023)
    HYBRID_EQUITY = "hybrid_equity"  # Treated as equity (≥65% equity allocation)
    HYBRID_DEBT = "hybrid_debt"  # Treated as debt (<65% equity allocation)


# Categories where equity taxation applies (≥65% equity allocation)
_EQUITY_TAX_CATEGORIES = {
    # Pure equity
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund",
    "Large & Mid Cap Fund", "Multi Cap Fund", "Flexi Cap Fund",
    "ELSS", "Sectoral/Thematic", "Value Fund", "Contra Fund",
    "Dividend Yield Fund", "Focused Fund", "Index Funds",
    "Other  ETFs",  # double space in AMFI data
    # Hybrid with ≥65% equity
    "Aggressive Hybrid Fund",
    "Balanced Advantage Fund",  # Dynamic — most maintain ≥65%
}

_DEBT_TAX_CATEGORIES = {
    "Liquid Fund", "Ultra Short Duration Fund", "Low Duration Fund",
    "Short Duration Fund", "Medium Duration Fund",
    "Medium to Long Duration Fund", "Long Duration Fund",
    "Dynamic Bond", "Corporate Bond Fund", "Credit Risk Fund",
    "Banking and PSU Fund", "Gilt Fund",
    "Gilt Fund with 10 year constant duration",
    "Money Market Fund", "Overnight Fund", "Floater Fund",
    # Hybrid with <65% equity
    "Conservative Hybrid Fund",
    "Equity Savings",  # 10-25% equity typically
}

_HYBRID_EQUITY_TAX_CATEGORIES = {
    "Multi Asset Allocation",  # Usually ≥65% equity+gold
}

_GOLD_TAX_CATEGORIES = {
    "Gold ETF", "Gold Fund",
}

_ELSS_CATEGORIES = {"ELSS"}


def _classify_tax_regime(holding: Holding) -> str:
    """Classify which tax regime applies to a holding."""
    # Direct equity/ETF → equity taxation
    if holding.asset_type == AssetType.EQUITY:
        return TaxRegime.EQUITY
    if holding.asset_type == AssetType.ETF:
        # Most ETFs (equity, gold) fall under equity taxation post-2023
        cat = holding.sector or ""
        if cat in _DEBT_TAX_CATEGORIES:
            return TaxRegime.DEBT
        return TaxRegime.EQUITY

    # Mutual funds — classify by AMFI category
    cat = holding.sector or ""

    if cat in _EQUITY_TAX_CATEGORIES:
        return TaxRegime.EQUITY
    if cat in _DEBT_TAX_CATEGORIES:
        return TaxRegime.DEBT
    if cat in _HYBRID_EQUITY_TAX_CATEGORIES:
        return TaxRegime.HYBRID_EQUITY
    if cat in _GOLD_TAX_CATEGORIES:
        return TaxRegime.EQUITY  # Gold ETFs/funds: equity taxation post July 2024

    # Fallback: check name keywords
    name_upper = (holding.scheme_name or "").upper()
    if any(kw in name_upper for kw in ("LIQUID", "DEBT", "BOND", "GILT", "MONEY MARKET", "OVERNIGHT")):
        return TaxRegime.DEBT
    if any(kw in name_upper for kw in ("GOLD", "SILVER")):
        return TaxRegime.EQUITY

    # Default unknown MFs to equity (most common)
    return TaxRegime.EQUITY


def _is_elss(holding: Holding) -> bool:
    """Check if holding is an ELSS fund (3-year lock-in)."""
    cat = holding.sector or ""
    if cat in _ELSS_CATEGORIES:
        return True
    name_upper = (holding.scheme_name or "").upper()
    return "ELSS" in name_upper or "TAX SAVER" in name_upper


# ---------------------------------------------------------------------------
# Tax rate constants (FY 2024-25, post July 23 2024 budget)
# ---------------------------------------------------------------------------

EQUITY_LTCG_RATE = 0.125  # 12.5%
EQUITY_LTCG_EXEMPTION = 125000  # ₹1.25L per year
EQUITY_STCG_RATE = 0.20  # 20%
DEBT_SLAB_RATE = 0.30  # Assume highest slab for worst-case


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class TaxRegimeBreakdown:
    """Gain/loss breakdown for a single tax regime."""
    regime: str  # "equity" | "debt" | "hybrid_equity" | "hybrid_debt"
    regime_label: str  # Human-readable label
    holdings_count: int
    total_value: float
    total_invested: float
    unrealized_gain: float
    unrealized_loss: float  # Absolute value of losses (positive number)
    net_gain_loss: float
    # Applicable tax info
    ltcg_rate_pct: float
    stcg_rate_pct: float
    tax_note: str  # Brief explanation of tax treatment


@dataclass
class HarvestCandidate:
    """A holding with unrealized loss — potential tax-loss harvest."""
    scheme_name: str
    isin: str | None
    asset_type: str
    tax_regime: str
    market_value: float
    invested_value: float
    unrealized_loss: float  # Negative number
    loss_pct: float  # Loss as % of invested


@dataclass
class ElssHolding:
    """An ELSS holding with lock-in awareness."""
    scheme_name: str
    isin: str | None
    market_value: float
    invested_value: float | None


@dataclass
class TaxSummary:
    """Summary of portfolio tax analysis."""
    total_unrealized_gain: float
    total_unrealized_loss: float
    net_unrealized: float
    # Worst-case tax estimates (if liquidated today)
    estimated_equity_tax: float  # On equity gains
    estimated_debt_tax: float  # On debt gains
    estimated_total_tax: float
    harvest_potential: float  # Total losses that could offset gains
    elss_count: int


@dataclass
class TaxResult:
    """Full output of the Tax Awareness Analyzer lens."""
    score: float  # 0-100, higher = more tax efficient

    summary: TaxSummary
    regime_breakdown: list[TaxRegimeBreakdown] = field(default_factory=list)
    harvest_candidates: list[HarvestCandidate] = field(default_factory=list)
    elss_holdings: list[ElssHolding] = field(default_factory=list)

    findings: list[str] = field(default_factory=list)
    holdings_analyzed: int = 0
    # Data limitation notice
    limitation_note: str = (
        "Tax estimates are approximate. Without transaction dates, we cannot "
        "determine whether gains are long-term or short-term. Shown figures "
        "assume worst-case (short-term) rates. Consult a tax professional "
        "for exact liability."
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_tax(portfolio: Portfolio) -> TaxResult:
    """Run the tax awareness analysis on a portfolio.

    Phase 1 lens — deterministic, no user input needed.
    """
    holdings = portfolio.holdings
    if not holdings:
        return TaxResult(
            score=50.0,
            summary=TaxSummary(
                total_unrealized_gain=0, total_unrealized_loss=0,
                net_unrealized=0, estimated_equity_tax=0,
                estimated_debt_tax=0, estimated_total_tax=0,
                harvest_potential=0, elss_count=0,
            ),
            holdings_analyzed=0,
        )

    # Classify each holding
    regime_data: dict[str, list[tuple[Holding, float, float]]] = {}
    harvest_candidates: list[HarvestCandidate] = []
    elss_holdings: list[ElssHolding] = []

    for h in holdings:
        mv = h.market_value or 0
        iv = h.invested_value or 0
        if mv <= 0 and iv <= 0:
            continue

        regime = _classify_tax_regime(h)

        if regime not in regime_data:
            regime_data[regime] = []
        regime_data[regime].append((h, mv, iv))

        # Tax-loss harvesting candidate
        if mv > 0 and iv > 0 and mv < iv:
            loss = mv - iv  # Negative
            loss_pct = loss / iv * 100
            harvest_candidates.append(HarvestCandidate(
                scheme_name=h.scheme_name,
                isin=h.isin,
                asset_type=h.asset_type.value,
                tax_regime=regime,
                market_value=mv,
                invested_value=iv,
                unrealized_loss=round(loss, 2),
                loss_pct=round(loss_pct, 1),
            ))

        # ELSS check
        if _is_elss(h):
            elss_holdings.append(ElssHolding(
                scheme_name=h.scheme_name,
                isin=h.isin,
                market_value=mv,
                invested_value=iv if iv > 0 else None,
            ))

    # Sort harvest candidates by loss (biggest loss first)
    harvest_candidates.sort(key=lambda c: c.unrealized_loss)

    # Build regime breakdown
    regime_labels = {
        TaxRegime.EQUITY: "Equity",
        TaxRegime.DEBT: "Debt",
        TaxRegime.HYBRID_EQUITY: "Hybrid (Equity-taxed)",
        TaxRegime.HYBRID_DEBT: "Hybrid (Debt-taxed)",
    }
    regime_tax_notes = {
        TaxRegime.EQUITY: (
            "LTCG (>12 months): 12.5% above ₹1.25L exemption. "
            "STCG (≤12 months): 20%."
        ),
        TaxRegime.DEBT: (
            "Gains taxed at income tax slab rate (no indexation benefit "
            "for MFs purchased after April 2023)."
        ),
        TaxRegime.HYBRID_EQUITY: (
            "Treated as equity for tax purposes (≥65% equity allocation). "
            "LTCG: 12.5% above ₹1.25L. STCG: 20%."
        ),
        TaxRegime.HYBRID_DEBT: (
            "Treated as debt for tax purposes (<65% equity allocation). "
            "Gains taxed at income tax slab rate."
        ),
    }
    regime_rates = {
        TaxRegime.EQUITY: (EQUITY_LTCG_RATE * 100, EQUITY_STCG_RATE * 100),
        TaxRegime.DEBT: (DEBT_SLAB_RATE * 100, DEBT_SLAB_RATE * 100),
        TaxRegime.HYBRID_EQUITY: (EQUITY_LTCG_RATE * 100, EQUITY_STCG_RATE * 100),
        TaxRegime.HYBRID_DEBT: (DEBT_SLAB_RATE * 100, DEBT_SLAB_RATE * 100),
    }

    breakdowns: list[TaxRegimeBreakdown] = []
    total_gain = 0.0
    total_loss = 0.0
    equity_gain = 0.0
    debt_gain = 0.0

    for regime, items in regime_data.items():
        total_value = sum(mv for _, mv, _ in items)
        total_invested = sum(iv for _, _, iv in items)
        gains = sum(mv - iv for _, mv, iv in items if mv > iv and iv > 0)
        losses = sum(iv - mv for _, mv, iv in items if mv < iv and iv > 0)
        net = gains - losses

        ltcg_rate, stcg_rate = regime_rates.get(regime, (12.5, 20.0))

        breakdowns.append(TaxRegimeBreakdown(
            regime=regime,
            regime_label=regime_labels.get(regime, regime),
            holdings_count=len(items),
            total_value=round(total_value, 2),
            total_invested=round(total_invested, 2),
            unrealized_gain=round(gains, 2),
            unrealized_loss=round(losses, 2),
            net_gain_loss=round(net, 2),
            ltcg_rate_pct=ltcg_rate,
            stcg_rate_pct=stcg_rate,
            tax_note=regime_tax_notes.get(regime, ""),
        ))

        total_gain += gains
        total_loss += losses

        if regime in (TaxRegime.EQUITY, TaxRegime.HYBRID_EQUITY):
            equity_gain += max(net, 0)
        else:
            debt_gain += max(net, 0)

    # Sort breakdowns by value (largest first)
    breakdowns.sort(key=lambda b: b.total_value, reverse=True)

    # Worst-case tax estimation (assume all STCG — no holding period data)
    equity_tax = max(equity_gain, 0) * EQUITY_STCG_RATE
    debt_tax = max(debt_gain, 0) * DEBT_SLAB_RATE
    total_tax = equity_tax + debt_tax

    harvest_potential = sum(abs(c.unrealized_loss) for c in harvest_candidates)

    summary = TaxSummary(
        total_unrealized_gain=round(total_gain, 2),
        total_unrealized_loss=round(total_loss, 2),
        net_unrealized=round(total_gain - total_loss, 2),
        estimated_equity_tax=round(equity_tax, 2),
        estimated_debt_tax=round(debt_tax, 2),
        estimated_total_tax=round(total_tax, 2),
        harvest_potential=round(harvest_potential, 2),
        elss_count=len(elss_holdings),
    )

    findings = _generate_findings(summary, breakdowns, harvest_candidates, elss_holdings)
    score = _compute_tax_score(summary, breakdowns, harvest_candidates)

    return TaxResult(
        score=score,
        summary=summary,
        regime_breakdown=breakdowns,
        harvest_candidates=harvest_candidates,
        elss_holdings=elss_holdings,
        findings=findings,
        holdings_analyzed=len(holdings),
    )


def _compute_tax_score(
    summary: TaxSummary,
    breakdowns: list[TaxRegimeBreakdown],
    harvest_candidates: list[HarvestCandidate],
) -> float:
    """Compute tax efficiency score (0-100, higher = more tax efficient).

    Rewards:
    - Having gains in equity regime (lower tax rates)
    - Awareness of harvest opportunities (losses to offset)
    - Having ELSS (Section 80C benefit)

    Penalizes:
    - Large unrealized gains in debt regime (highest tax rate)
    - No diversification across tax regimes
    - Large potential tax liability relative to portfolio
    """
    if summary.net_unrealized <= 0:
        # Portfolio at net loss — no tax liability, high score
        return 90.0

    penalty = 0.0

    # Debt-heavy gains penalty (0-30 points)
    # Gains in debt are taxed at slab rate (~30%) vs equity (~12.5-20%)
    total_gains = summary.total_unrealized_gain
    if total_gains > 0:
        debt_gain_ratio = sum(
            max(b.net_gain_loss, 0) for b in breakdowns
            if b.regime in (TaxRegime.DEBT, TaxRegime.HYBRID_DEBT)
        ) / total_gains
        penalty += debt_gain_ratio * 30

    # Tax liability relative to portfolio (0-25 points)
    total_value = sum(b.total_value for b in breakdowns)
    if total_value > 0:
        tax_ratio = summary.estimated_total_tax / total_value
        # 5% tax/value ratio = full penalty
        penalty += min(tax_ratio / 0.05 * 25, 25)

    # Unharvested losses penalty (0-15 points)
    # If you have losses you could harvest but also have gains, penalty
    if summary.harvest_potential > 0 and summary.total_unrealized_gain > 0:
        # More unharvested losses relative to gains = higher penalty
        harvest_ratio = min(summary.harvest_potential / summary.total_unrealized_gain, 1)
        penalty += harvest_ratio * 15

    # Single tax regime penalty (0-10 points)
    regimes_with_value = sum(1 for b in breakdowns if b.total_value > 0)
    if regimes_with_value <= 1:
        penalty += 10

    score = round(max(100 - penalty, 0), 1)
    return min(score, 100.0)


def _generate_findings(
    summary: TaxSummary,
    breakdowns: list[TaxRegimeBreakdown],
    harvest_candidates: list[HarvestCandidate],
    elss_holdings: list[ElssHolding],
) -> list[str]:
    """Generate plain-English findings about tax implications."""
    findings: list[str] = []

    # Net gain/loss summary
    if summary.net_unrealized > 0:
        findings.append(
            f"Your portfolio has ₹{summary.total_unrealized_gain:,.0f} in unrealized "
            f"gains and ₹{summary.total_unrealized_loss:,.0f} in unrealized losses "
            f"(net: ₹{summary.net_unrealized:,.0f}). Tax applies only when you sell."
        )
    elif summary.net_unrealized < 0:
        findings.append(
            f"Your portfolio is currently at a net unrealized loss of "
            f"₹{abs(summary.net_unrealized):,.0f}. No tax liability on unrealized losses."
        )

    # Regime-specific findings
    for b in breakdowns:
        if b.net_gain_loss > 0 and b.regime == TaxRegime.DEBT:
            findings.append(
                f"₹{b.net_gain_loss:,.0f} of your gains are in debt instruments, "
                f"taxed at your income slab rate (up to 30%). Debt MFs purchased "
                f"after April 2023 no longer receive indexation benefits."
            )

    # Tax-loss harvesting
    if harvest_candidates and summary.total_unrealized_gain > 0:
        total_harvestable = sum(abs(c.unrealized_loss) for c in harvest_candidates)
        findings.append(
            f"{len(harvest_candidates)} holding{'s' if len(harvest_candidates) > 1 else ''} "
            f"{'are' if len(harvest_candidates) > 1 else 'is'} currently at a loss "
            f"(total: ₹{total_harvestable:,.0f}). Under Indian tax rules, realized "
            f"losses can be set off against realized gains in the same financial year."
        )

    # ELSS
    if elss_holdings:
        total_elss = sum(e.market_value for e in elss_holdings)
        findings.append(
            f"{len(elss_holdings)} ELSS holding{'s' if len(elss_holdings) > 1 else ''} "
            f"(₹{total_elss:,.0f}) provide Section 80C tax deduction up to ₹1.5L/year. "
            f"ELSS has a mandatory 3-year lock-in period."
        )

    # Worst-case tax estimate
    if summary.estimated_total_tax > 0:
        findings.append(
            f"If you were to sell all holdings at current prices, estimated worst-case "
            f"tax would be approximately ₹{summary.estimated_total_tax:,.0f}. Actual "
            f"tax depends on your holding period and income slab."
        )

    return findings
