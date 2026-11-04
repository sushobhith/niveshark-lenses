# niveshark-lenses

The scoring math behind Niveshark's portfolio health checkup, and the
scrubber that removes personal details from a statement before anything
leaves the browser. These are the same files Niveshark runs in production;
a check in Niveshark's private repository fails its CI if the two copies
differ by anything other than import lines.

No model, no network, no database: these files import only the standard
library, pydantic and each other.

Version: `LENSES_VERSION = "1.0.0"` (in `lenses/__init__.py`). The
Niveshark site footer shows the same number.

## What each lens measures

| File | Lens | What it measures |
|---|---|---|
| `lenses/diversification.py` | Diversification | Spread across three dimensions: asset class (equity, debt, hybrid, gold, international, cash), scored by HHI anchored to three classes; market-cap mix, scored as 100 × (1 − L1(w, market) / 2) against a 70/20/10 large/mid/small reference (approximately the Nifty 500); and concentration across the AMFI categories held. |
| `lenses/overlap.py` | Overlap | Stocks shared across mutual funds, ETFs and directly held shares. Pairwise overlap is a weighted Jaccard similarity on stock-weight vectors; a fund pair is reported from 15%, moderate from 25% and high from 40% (`_OVERLAP_*_THRESHOLD` in the file). |
| `lenses/cost.py` | Cost Efficiency | Regular-plan mutual fund holdings that have a direct-plan counterpart, the annual expense-ratio difference between the two, and high-expense holdings. Expense differentials are category estimates. |
| `lenses/risk.py` | Risk Profile | Concentration at the holding level: single holdings above 10% of the portfolio, the share of the top 3/5/10 holdings, a volatility bucket from asset composition, and an overall risk level. Concentration penalties apply to concentrated instruments (direct equity, sectoral/thematic funds, uncategorised funds); diversified funds carry a smaller single-manager penalty. No historical price data. |
| `lenses/tax.py` | Tax Awareness | Each holding's tax regime (equity, debt, hybrid), unrealised gains and losses by regime, loss-harvesting candidates, ELSS lock-in, and the worst-case tax if everything were sold today, under the rules applicable from July 2024 (equity LTCG 12.5% above ₹1.25 lakh, STCG 20%; debt funds bought after April 2023 at slab rate). |
| `lenses/rupee_cost.py` | Rupee cost | The rupee figures from the cost, overlap and tax lenses in one place: annual regular-vs-direct expense difference, annual duplicate expense on overlapping fund pairs, and the one-time worst-case tax. Not part of the score. |
| `lenses/health_score.py` | Health score | The 0–100 composite of the five lens scores. |

Supporting files: `lenses/models.py` (the holding and portfolio models) and
`lenses/scheme_master.py` (in-memory lookups over the AMFI scheme list that
the cost lens uses to tell direct plans from regular ones).

## How the health score is computed

Weights, from `DEFAULT_WEIGHTS` in `lenses/health_score.py` (they sum to 1.0):

| Lens | Weight |
|---|---|
| diversification | 0.25 |
| risk | 0.25 |
| overlap | 0.20 |
| cost | 0.15 |
| tax | 0.15 |

When a lens has no data for a portfolio, the remaining weights are scaled
up proportionally so they still sum to 1.0.

Grades, from `_score_to_grade`:

| Score | Grade |
|---|---|
| 80 and above | A |
| 65 to below 80 | B |
| 50 to below 65 | C |
| 35 to below 50 | D |
| below 35 | F |

## The scrubber

`scrub/scrub.ts` runs in the browser between reading a statement PDF and
parsing it. It drops the investor block above the first holdings table
(keeping only the statement date), redacts PAN, email, IFSC and date-of-birth
values, and drops rows that start with a personal-details label. On the way
out, folio numbers are hashed with a per-session salt, client ids are
dropped, every holding is copied through a field allowlist, and the
serialised payload is checked once more for PAN, email, IFSC or any value
that was redacted; if one is found, nothing is sent.

`scrub/test-support.ts` exists only for the tests: Niveshark's statement
parsers are not in this repository, so it reads the invented statements in
`scrub.test.ts` into holdings with a minimal row reader. The scrubbing,
hashing, allowlist and leak check under test are the real `scrub.ts`.

## Running the tests

Python (3.10 or newer):

```sh
pip install -r requirements.txt
python3 -m pytest -q
```

TypeScript (Node 22.6 or newer, no dependencies to install):

```sh
npm test
```

## License

MIT. See `LICENSE`.
