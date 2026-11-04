"""Holding models the lenses read.

`AssetType`, `HoldingMode`, `Holding` and `Portfolio` are Niveshark's
`schemas/holdings.py` (without the parser-only `ParseResult`);
`UnderlyingHolding` and `SchemePortfolio` are the two dataclasses from
`ingestion/holdings_provider.py` that the overlap lens consumes.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    EQUITY = "equity"


class HoldingMode(str, Enum):
    SOA = "soa"       # Statement of Account (non-demat MF)
    DEMAT = "demat"   # Demat (MF/ETF/equity held via depository)


class Holding(BaseModel):
    """A single holding in a user's portfolio, normalized across all sources."""

    scheme_name: str
    isin: Optional[str] = None
    symbol: Optional[str] = None  # BSE/NSE trading symbol (for equities)

    asset_type: AssetType
    holding_mode: Optional[HoldingMode] = None

    folio_number: Optional[str] = None    # MF folio (SoA holdings)
    client_id: Optional[str] = None       # Demat client ID

    units: float
    avg_cost_price: Optional[float] = None  # Per-unit cost
    invested_value: Optional[float] = None  # Total cost basis
    nav: Optional[float] = None             # Latest NAV or price
    nav_date: Optional[date] = None
    market_value: Optional[float] = None    # Current value

    sector: Optional[str] = None
    registrar: Optional[str] = None  # CAMS / KFINTECH

    source: str  # e.g. "mfcentral", "cams", "zerodha"

    @property
    def dedup_keys(self) -> list[str]:
        """All possible dedup keys for this holding.

        The merger checks if ANY key matches an existing holding.
        Multiple keys allow matching across sources that have different
        identifiers for the same holding (e.g., folio vs ISIN vs name).
        """
        keys: list[str] = []
        if self.folio_number:
            keys.append(f"folio:{self.folio_number}")
        if self.isin:
            keys.append(f"isin:{self.isin}")
        keys.append(f"name:{_normalize_name(self.scheme_name)}")
        return keys


class Portfolio(BaseModel):
    """Merged, deduplicated portfolio from all uploaded files."""

    holdings: list[Holding]
    investor_name: Optional[str] = None
    investor_pan: Optional[str] = None
    sources_used: list[str] = Field(default_factory=list)
    duplicates_removed: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_invested(self) -> float:
        return sum(h.invested_value or 0 for h in self.holdings)

    @property
    def total_market_value(self) -> float:
        return sum(h.market_value or 0 for h in self.holdings)

    @property
    def mf_count(self) -> int:
        return sum(1 for h in self.holdings if h.asset_type == AssetType.MUTUAL_FUND)

    @property
    def etf_count(self) -> int:
        return sum(1 for h in self.holdings if h.asset_type == AssetType.ETF)

    @property
    def equity_count(self) -> int:
        return sum(1 for h in self.holdings if h.asset_type == AssetType.EQUITY)


def _normalize_name(name: str) -> str:
    """Normalize scheme/stock name for fuzzy dedup matching."""
    import re

    name = name.upper().strip()
    # Remove common noise words and punctuation
    noise = [
        "MUTUAL FUND", "MF", "DIRECT", "REGULAR", "GROWTH", "PLAN",
        "DIR", "REG", "PL", "-", "(", ")", ".", ",",
    ]
    for word in noise:
        name = name.replace(word, "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Fund composition (from ingestion/holdings_provider.py)
# ---------------------------------------------------------------------------

@dataclass
class UnderlyingHolding:
    """A single stock/bond held by a mutual fund or ETF."""
    security_name: str
    isin: str | None = None
    weight: float = 0.0  # percentage weight (0-100)
    sector: str | None = None
    market_value: float | None = None


@dataclass
class SchemePortfolio:
    """Portfolio composition of a mutual fund or ETF."""
    scheme_isin: str
    scheme_name: str
    provider_id: str = ""  # provider-specific identifier (family_id, morningstar_id, etc.)
    holdings: list[UnderlyingHolding] = field(default_factory=list)
    total_holdings_count: int = 0
    fetched_at: float = 0.0  # timestamp
