/**
 * Test support for `scrub.test.ts`. Not part of the scrubber.
 *
 * Niveshark's test drives the scrubber through the full browser pipeline
 * (detect → scrub → parse → finalise → payload). The statement parsers are
 * not in this repository, so this file stands in for the two pipeline
 * functions the test imports, with a minimal reader for the invented
 * statements the test builds:
 *
 *   - the source is read from the file name ("cams" → CAMS, else MFCentral);
 *   - the document goes through `scrubRows` exactly as in production;
 *   - every scrubbed row whose first cell is a folio number becomes one
 *     holding (MFCentral: folio, scheme, invested, units, NAV date, NAV,
 *     value; CAMS: folio, ISIN, scheme, cost, units, NAV date, NAV, value,
 *     registrar);
 *   - the holdings go through `finaliseHoldings` with the session salt.
 *
 * The scrubbing, hashing, allowlist and leak assertion under test are the
 * real `scrub.ts`; only the row-to-holding step is simplified.
 */

import { finaliseHoldings, scrubRows, sessionSalt } from "./scrub.ts";
import type { ClientHolding, ClientParseResult, ExtractedDocument, StatementSource } from "./types.ts";

const FOLIO_RE = /^\d{6,}(\/\d+)?$/;

const redactedByResult = new WeakMap<ClientParseResult, readonly string[]>();

export function redactedValuesFor(result: ClientParseResult): readonly string[] {
  return redactedByResult.get(result) ?? [];
}

function num(text: string): number | null {
  const n = Number(text.replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

function holding(source: StatementSource, cells: string[]): ClientHolding | null {
  const base = {
    symbol: null,
    asset_type: "mutual_fund" as const,
    holding_mode: "soa" as const,
    avg_cost_price: null,
    sector: null,
    plan_type: null,
    source,
  };
  if (source === "cams" && cells.length >= 9) {
    const [folio, isin, scheme, cost, units, navDate, nav, value, registrar] = cells;
    return {
      ...base,
      scheme_name: scheme.replace(/^\w+ - /, ""),
      isin,
      folio_number: folio,
      units: num(units) ?? 0,
      invested_value: num(cost),
      nav: num(nav),
      nav_date: navDate,
      market_value: num(value),
      registrar,
    };
  }
  if (source === "mfcentral" && cells.length >= 7) {
    const [folio, scheme, invested, units, navDate, nav, value] = cells;
    return {
      ...base,
      scheme_name: scheme,
      isin: null,
      folio_number: folio,
      units: num(units) ?? 0,
      invested_value: num(invested),
      nav: num(nav),
      nav_date: navDate,
      market_value: num(value),
      registrar: null,
    };
  }
  return null;
}

export async function parseExtractedDocument(doc: ExtractedDocument, filename: string): Promise<ClientParseResult> {
  const source: StatementSource = /cams/i.test(filename) ? "cams" : "mfcentral";
  const scrubbed = scrubRows(doc, source);
  const holdings: ClientHolding[] = [];
  for (const page of scrubbed.doc.pages) {
    for (const row of page.rows) {
      const cells = row.cells.map((c) => c.text);
      if (!FOLIO_RE.test(cells[0] ?? "")) continue;
      const h = holding(source, cells);
      if (h) holdings.push(h);
    }
  }
  const parsed: ClientParseResult = { source, filename, holdings, warnings: [], statement_date: null };
  const finalised = await finaliseHoldings(parsed, sessionSalt);
  redactedByResult.set(finalised, scrubbed.redacted);
  return finalised;
}
