/**
 * Types for the statement scrubber: the subset of Niveshark's
 * `app/frontend/lib/cas/types.ts` that `scrub.ts` and its test use.
 *
 * Erasable TypeScript only: runs unmodified under
 * `node --experimental-strip-types`.
 */

// ---------------------------------------------------------------------------
// Extraction (D-024)
// ---------------------------------------------------------------------------

export interface TextCell {
  x: number;
  width: number;
  text: string;
}

export interface TextRow {
  y: number;
  cells: TextCell[];
  text: string;
}

export interface ExtractedPage {
  page: number;
  rows: TextRow[];
}

export interface ExtractedDocument {
  pages: ExtractedPage[];
}

// ---------------------------------------------------------------------------
// Parsing (D-025)
// ---------------------------------------------------------------------------

export type StatementSource = "mfcentral" | "cams" | "zerodha";

export type ClientAssetType = "mutual_fund" | "etf" | "equity";
export type ClientHoldingMode = "soa" | "demat";

/** The frontend `Holding` shape plus `registrar`. */
export interface ClientHolding {
  scheme_name: string;
  isin: string | null;
  symbol: string | null;
  asset_type: ClientAssetType;
  holding_mode: ClientHoldingMode | null;
  folio_number: string | null;
  units: number;
  avg_cost_price: number | null;
  invested_value: number | null;
  nav: number | null;
  nav_date: string | null;
  market_value: number | null;
  sector: string | null;
  plan_type: string | null;
  source: string;
  registrar: string | null;
  /** Parser-internal only. Dropped by `finaliseHoldings`, never serialised. */
  client_id?: string | null;
}

/** A holding after `finaliseHoldings`: hashed folio, no client id. */
export type PayloadHolding = Omit<ClientHolding, "client_id">;

export interface ClientParseResult {
  source: StatementSource;
  filename: string;
  holdings: ClientHolding[];
  warnings: string[];
  statement_date: string | null;
}


export interface AnalyzeSource {
  source: string;
  holdings: PayloadHolding[];
}

export interface AnalyzePayload {
  sources: AnalyzeSource[];
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** Thrown when a serialised payload still contains a PAN, email, IFSC or a redacted value. */
export class PiiLeakError extends Error {
  /** Which rule fired; never the matched value itself. */
  rule: string;
  constructor(rule: string) {
    super(`Payload failed the PII assertion (${rule})`);
    this.name = "PiiLeakError";
    this.rule = rule;
  }
}
