"""Scheme-master lookups the cost lens calls.

`SchemeInfo`, `AMFIMaster` and its name normaliser are from Niveshark's
`ingestion/amfi.py` (the dataclasses and lookup methods only; downloading
and parsing AMFI's NAVAll.txt stay out). `classify_direct_regular`,
`find_direct_alternative` and `_strip_plan_type` are from
`ingestion/enrichment.py`. Deterministic lookups over an in-memory scheme
list: no network, no database.
"""

import re
from dataclasses import dataclass, field

from lenses.models import Holding


@dataclass
class SchemeInfo:
    """A single mutual fund scheme from the AMFI master."""

    scheme_code: int
    isin_growth: str | None  # ISIN for growth/payout option
    isin_reinvest: str | None  # ISIN for dividend reinvestment option
    scheme_name: str
    nav: float | None
    nav_date: str | None
    amc: str | None = None
    category: str | None = None
    is_direct: bool = False
    is_regular: bool = False


@dataclass
class AMFIMaster:
    """Indexed AMFI scheme master for fast lookups."""

    schemes: list[SchemeInfo] = field(default_factory=list)
    _by_isin: dict[str, SchemeInfo] = field(default_factory=dict, repr=False)
    _by_code: dict[int, SchemeInfo] = field(default_factory=dict, repr=False)
    _by_name_normalized: dict[str, SchemeInfo] = field(default_factory=dict, repr=False)

    def build_index(self) -> None:
        """Build lookup indices after schemes are populated."""
        self._by_isin.clear()
        self._by_code.clear()
        self._by_name_normalized.clear()

        for s in self.schemes:
            if s.isin_growth and s.isin_growth != "-":
                self._by_isin[s.isin_growth] = s
            if s.isin_reinvest and s.isin_reinvest != "-":
                self._by_isin[s.isin_reinvest] = s
            self._by_code[s.scheme_code] = s
            norm = _normalize_scheme_name(s.scheme_name)
            self._by_name_normalized[norm] = s

    def lookup_by_isin(self, isin: str) -> SchemeInfo | None:
        return self._by_isin.get(isin.strip().upper()) if isin else None

    def lookup_by_code(self, code: int) -> SchemeInfo | None:
        return self._by_code.get(code)

    def lookup_by_name(self, name: str) -> SchemeInfo | None:
        """Exact match on normalized name."""
        norm = _normalize_scheme_name(name)
        return self._by_name_normalized.get(norm)

    def search_by_name(self, name: str, max_results: int = 5) -> list[SchemeInfo]:
        """Fuzzy search by scheme name. Returns best matches."""
        query = _normalize_scheme_name(name)
        if not query:
            return []

        # Try exact match first
        exact = self._by_name_normalized.get(query)
        if exact:
            return [exact]

        # Token-based matching
        query_tokens = set(query.split())
        scored: list[tuple[float, SchemeInfo]] = []

        for norm_name, scheme in self._by_name_normalized.items():
            name_tokens = set(norm_name.split())
            if not query_tokens:
                continue
            overlap = len(query_tokens & name_tokens)
            score = overlap / max(len(query_tokens), len(name_tokens))
            if score >= 0.5:
                scored.append((score, scheme))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_results]]

    @property
    def scheme_count(self) -> int:
        return len(self.schemes)


def _normalize_scheme_name(name: str) -> str:
    """Normalize scheme name for matching. Strips noise, lowercases."""
    name = name.upper().strip()
    # Remove common suffixes and noise
    noise = [
        "MUTUAL FUND", "- DIRECT", "DIRECT", "- REGULAR", "REGULAR",
        "- GROWTH", "GROWTH", "- IDCW", "IDCW",
        "- MONTHLY", "MONTHLY", "- QUARTERLY", "QUARTERLY",
        "- ANNUAL", "ANNUAL", "PLAN", "OPTION",
        "- DIV", "DIVIDEND", "PAYOUT", "REINVESTMENT",
        "(", ")", "-", ".", ",",
    ]
    for word in noise:
        name = name.replace(word, "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def classify_direct_regular(holding: Holding, master: AMFIMaster) -> str | None:
    """Classify whether a holding is Direct or Regular plan.

    Returns "direct", "regular", or None if unknown.
    """
    # Check AMFI master first
    scheme = None
    if holding.isin:
        scheme = master.lookup_by_isin(holding.isin)
    if not scheme:
        matches = master.search_by_name(holding.scheme_name)
        if matches:
            scheme = matches[0]

    if scheme:
        if scheme.is_direct:
            return "direct"
        if scheme.is_regular:
            return "regular"

    # Fallback: check scheme name keywords
    name_upper = holding.scheme_name.upper()
    if "DIRECT" in name_upper or "- DIR" in name_upper:
        return "direct"
    if "REGULAR" in name_upper or "- REG" in name_upper:
        return "regular"

    return None


def find_direct_alternative(
    holding: Holding, master: AMFIMaster
) -> SchemeInfo | None:
    """For a regular-plan holding, find its direct-plan counterpart."""
    scheme = None
    if holding.isin:
        scheme = master.lookup_by_isin(holding.isin)
    if not scheme:
        matches = master.search_by_name(holding.scheme_name)
        scheme = matches[0] if matches else None

    if not scheme or scheme.is_direct:
        return None  # Already direct or unknown

    # Search for matching direct plan by normalized name
    base_name = _strip_plan_type(scheme.scheme_name)
    candidates = master.search_by_name(base_name + " DIRECT GROWTH", max_results=3)

    for candidate in candidates:
        if candidate.is_direct and candidate.amc == scheme.amc:
            return candidate

    return None


def _strip_plan_type(name: str) -> str:
    """Remove Direct/Regular/Growth/IDCW suffixes to find base scheme name."""
    import re
    name = re.sub(
        r"\s*-?\s*(DIRECT|REGULAR|RETAIL|GROWTH|IDCW|DIVIDEND|PAYOUT|REINVESTMENT)\s*",
        " ",
        name,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", name).strip()
