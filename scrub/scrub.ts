/**
 * Statement scrubber (D-003, D-026).
 *
 * Text layer (`scrubRows`), between detection and parsing:
 *   1. drop every row above the first table header row of a page, except
 *      the statement-date ("as on") cells;
 *   2. redact PAN, email, IFSC and DOB values in surviving rows;
 *   3. drop rows that start with a PII label, wherever they appear.
 * Depository statements (D-051) use their own table-header boundary and,
 * before rule 2, drop labelled DP ID / Client ID / BO ID / CAS ID rows and
 * redact every learnt identifier and the investor's name in every row.
 *
 * Payload layer: `finaliseHoldings` hashes folios with the session salt
 * and drops client ids; `toPayloadHolding` copies only allowlisted fields;
 * `buildAnalyzePayload` serialises through the allowlist and then
 * `assertNoPii` refuses to return a payload that still carries PII.
 */

import type {
  AnalyzePayload,
  ClientHolding,
  ClientParseResult,
  ExtractedDocument,
  ExtractedPage,
  PayloadHolding,
  StatementSource,
  TextCell,
  TextRow,
} from "./types.ts";
import { PiiLeakError } from "./types.ts";

export const PAN_RE = /[A-Z]{5}[0-9]{4}[A-Z]/g;
export const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
export const IFSC_RE = /\b[A-Z]{4}0[A-Z0-9]{6}\b/g;
const DOB_RE = /\b(DOB|Date of Birth)\b.*/i;
const PII_LABEL_RE =
  /^(name|investor|address|e-?mail|mobile|phone|contact|bank|account|a\/c|ifsc|nominee|kyc|date of birth|dob|pan)\b/i;
/**
 * Labels that are also ordinary words at the start of a wrapped scheme-name
 * line ("Bank BeES", "Nifty Bank Index Fund" wrapped at "Bank"). A row that
 * starts with one of these is a PII row only if it also carries a colon or a
 * digit (D-026 rule 3, tightened 2026-11-04).
 */
const AMBIGUOUS_LABEL_RE = /^(bank|account|contact|name)\b/i;

export function isPiiLabelRow(text: string): boolean {
  if (!PII_LABEL_RE.test(text)) return false;
  if (AMBIGUOUS_LABEL_RE.test(text) && !/[:\d]/.test(text)) return false;
  return true;
}
const AS_ON_RE = /as on/i;
const DATE_CELL_RE = /^\(?\s*\d{2}-[A-Za-z]{3}-\d{4}\s*\)?$/;

/** A table header row: `Folio No.`, `Client Id` or `ISIN` together with `Scheme`. */
export function isTableHeaderRow(row: TextRow): boolean {
  return /Folio No\.|Client Id|ISIN/.test(row.text) && /Scheme/.test(row.text);
}

/**
 * A depository holdings-table header (D-051): `ISIN` with `Security` (demat
 * table) or `Scheme Name` (folio table), and no colon. The colon rule keeps
 * labelled lines such as "Scheme Name : … ISIN : …" in the per-folio
 * descriptive blocks from ending the preamble early.
 */
export function isDepositoryTableHeaderRow(row: TextRow): boolean {
  return /ISIN/.test(row.text) && /Security|Scheme Name/i.test(row.text) && !/:/.test(row.text);
}

function headerPredicate(source: StatementSource): (row: TextRow) => boolean {
  return source === "depository" ? isDepositoryTableHeaderRow : isTableHeaderRow;
}

// ---------------------------------------------------------------------------
// Depository identifiers (D-051, SB-14..SB-17)
//
// Patterns follow casparser 1.4.1 `parsers/cdsl.py` (`SUMMARY_DPC_RE`,
// `SECTION_DPC_RE`, `SECTION_BOID_RE`, `PAN_RE`) and `parsers/_investor.py`
// (`_ID_MARKER_RE`: the investor's name is the first line after "CAS ID :" /
// "NSDL ID :"). The values are learnt from the unscrubbed rows of every page,
// then redacted wherever they recur, so a copy after the first holdings table
// (a second account header, a page banner, a footer) cannot survive.
// ---------------------------------------------------------------------------

const DP_ID_LEARN_RE = /\bDP\s*Id\s*:\s*([A-Z0-9]{4,})/gi;
const CLIENT_ID_LEARN_RE = /\bClient\s*Id\s*:\s*([A-Z0-9]{4,})/gi;
const BO_ID_LEARN_RE = /\b(?:BO\s*ID|DPID)\s*:\s*([A-Z0-9]{16})\b/gi;
const CAS_ID_LEARN_RE = /\b(?:CAS|NSDL)\s*ID\s*:\s*([A-Z0-9]{4,})/gi;
const OWNER_PAN_RE = /^(.+?)\s*\(\s*PAN\s*:/i;
const ID_MARKER_RE = /^\s*(?:CAS|NSDL)\s*ID\s*:/i;
/** Any 16-character BO ID shape: 16 digits (CDSL) or IN + 14 digits (NSDL). */
const BO_ID_SHAPE_RE = /\b(?:\d{16}|IN\d{14})\b/g;
/** A labelled depository-identifier row; dropped wherever it appears. */
const DEPOSITORY_ID_LABEL_RE = /\b(?:DP\s*Name|DP\s*Id|Client\s*Id|BO\s*ID|DPID|CAS\s*ID|NSDL\s*ID)\s*:/i;

interface LearntValue {
  value: string;
  token: string;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Every DP ID, Client ID, BO ID, CAS ID and investor name the statement prints. */
export function learnDepositoryIdentifiers(doc: ExtractedDocument): LearntValue[] {
  const found = new Map<string, string>();
  const add = (value: string | undefined, token: string) => {
    const v = value?.trim();
    if (v && v.length >= 4 && !found.has(v)) found.set(v, token);
  };
  for (const page of doc.pages) {
    page.rows.forEach((row, i) => {
      const t = row.text;
      for (const m of t.matchAll(BO_ID_LEARN_RE)) {
        add(m[1], "[BO_ID]");
        add(m[1].slice(0, 8), "[DP_ID]");
        add(m[1].slice(8), "[CLIENT_ID]");
      }
      for (const m of t.matchAll(DP_ID_LEARN_RE)) add(m[1], "[DP_ID]");
      for (const m of t.matchAll(CLIENT_ID_LEARN_RE)) add(m[1], "[CLIENT_ID]");
      for (const m of t.matchAll(CAS_ID_LEARN_RE)) add(m[1], "[CAS_ID]");
      for (const cell of row.cells) {
        const owner = OWNER_PAN_RE.exec(cell.text);
        if (owner) add(owner[1], "[NAME]");
      }
      if (ID_MARKER_RE.test(row.cells[0]?.text ?? "")) {
        const next = page.rows[i + 1]?.cells[0]?.text;
        if (next && !/[:\d]/.test(next)) add(next, "[NAME]");
      }
    });
  }
  // Longest first, so a BO ID is replaced whole before its DP ID half.
  return [...found].map(([value, token]) => ({ value, token })).sort((a, b) => b.value.length - a.value.length);
}

function redactLearnt(text: string, learnt: readonly LearntValue[], redacted: string[]): string {
  let out = text;
  for (const { value, token } of learnt) {
    const re = new RegExp(`(?<![A-Za-z0-9])${escapeRe(value)}(?![A-Za-z0-9])`, "gi");
    out = out.replace(re, (m) => {
      redacted.push(m);
      return token;
    });
  }
  return out.replace(BO_ID_SHAPE_RE, (m) => {
    redacted.push(m);
    return "[BO_ID]";
  });
}

function redactDepositoryRow(row: TextRow, learnt: readonly LearntValue[], redacted: string[]): TextRow | null {
  if (DEPOSITORY_ID_LABEL_RE.test(row.text)) return null;
  const cells = row.cells
    .map((c) => ({ ...c, text: redactLearnt(c.text, learnt, redacted) }))
    // A cell that was nothing but an identifier (a page banner with the
    // investor's name) goes, so it cannot be joined into a wrapped name.
    .filter((c) => !/^(?:\[[A-Z_]+\]\s*)+$/.test(c.text.trim()));
  return cells.length ? makeRow(row.y, cells) : null;
}

function makeRow(y: number, cells: TextCell[]): TextRow {
  return { y, cells, text: cells.map((c) => c.text).join(" ") };
}

/**
 * Keep only the statement-date cells of a preamble row. In two-column
 * preambles a name or address can share a y with an "as on" sentence.
 */
function keepAsOnCells(row: TextRow): TextRow | null {
  if (!AS_ON_RE.test(row.text)) return null;
  const cells = row.cells.filter((c) => AS_ON_RE.test(c.text) || DATE_CELL_RE.test(c.text));
  return cells.length ? makeRow(row.y, cells) : null;
}

function redactCell(text: string, redacted: string[]): string {
  const replace = (re: RegExp, token: string) => (s: string) =>
    s.replace(re, (m) => {
      redacted.push(m);
      return token;
    });
  let out = text;
  out = replace(PAN_RE, "[PAN]")(out);
  out = replace(EMAIL_RE, "[EMAIL]")(out);
  out = replace(IFSC_RE, "[IFSC]")(out);
  return out;
}

/** Redact values in one row. A DOB label redacts the rest of the row, across cells. */
function redactRow(row: TextRow, redacted: string[]): TextRow {
  const cells: TextCell[] = [];
  let dobSeen = false;
  for (const cell of row.cells) {
    if (dobSeen) {
      redacted.push(cell.text);
      continue;
    }
    let text = redactCell(cell.text, redacted);
    const dob = DOB_RE.exec(text);
    if (dob) {
      redacted.push(dob[0]);
      text = text.slice(0, dob.index) + "[DOB]";
      dobSeen = true;
    }
    if (text.trim() !== "") cells.push({ ...cell, text });
  }
  return makeRow(row.y, cells);
}

function scrubPage(
  page: ExtractedPage,
  headerSeenBefore: boolean,
  redacted: string[],
  isHeader: (row: TextRow) => boolean,
  preRedact: ((row: TextRow) => TextRow | null) | null,
): { page: ExtractedPage; headerSeen: boolean } {
  const headerIdx = page.rows.findIndex(isHeader);
  // A page with no header is preamble until the first table has been seen,
  // and a table continuation afterwards.
  const keepFrom = headerIdx >= 0 ? headerIdx : headerSeenBefore ? 0 : page.rows.length;

  const rows: TextRow[] = [];
  page.rows.forEach((row, i) => {
    let candidate: TextRow | null = row;
    if (i < keepFrom) candidate = keepAsOnCells(row);
    if (candidate && preRedact) candidate = preRedact(candidate);
    if (!candidate) return;
    if (isPiiLabelRow(candidate.text)) return;
    const clean = redactRow(candidate, redacted);
    if (clean.cells.length === 0) return;
    if (isPiiLabelRow(clean.text)) return;
    rows.push(clean);
  });

  return { page: { page: page.page, rows }, headerSeen: headerSeenBefore || headerIdx >= 0 };
}

/**
 * Text-layer scrub. `redacted` holds the raw values that were replaced, for
 * the leak assertion only; it stays in memory and is never serialised.
 */
export function scrubRows(
  doc: ExtractedDocument,
  detectedSource: StatementSource,
): { doc: ExtractedDocument; redacted: string[] } {
  const isHeader = headerPredicate(detectedSource);
  const redacted: string[] = [];
  // Depository statements (D-051): learn the account identifiers and the
  // investor's name from every page first, redact them in every surviving
  // row, and hand all of them to the leak assertion even if no copy survived.
  const learnt = detectedSource === "depository" ? learnDepositoryIdentifiers(doc) : [];
  const preRedact = detectedSource === "depository" ? (row: TextRow) => redactDepositoryRow(row, learnt, redacted) : null;
  const pages: ExtractedPage[] = [];
  let headerSeen = false;
  for (const page of doc.pages) {
    const out = scrubPage(page, headerSeen, redacted, isHeader, preRedact);
    headerSeen = out.headerSeen;
    pages.push(out.page);
  }
  for (const { value } of learnt) redacted.push(value);
  return { doc: { pages }, redacted };
}

// ---------------------------------------------------------------------------
// Payload layer
// ---------------------------------------------------------------------------

function randomHex(bytes: number): string {
  const buf = new Uint8Array(bytes);
  globalThis.crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** 16 random bytes, hex; created once per module load and never written anywhere. */
export const sessionSalt: string = randomHex(16);

export async function hashFolio(folio: string, salt: string): Promise<string> {
  const data = new TextEncoder().encode(folio + salt);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
}

/** Copy only the allowlisted fields. Anything else a parser sets is not serialised. */
export function toPayloadHolding(h: ClientHolding | PayloadHolding): PayloadHolding {
  return {
    scheme_name: h.scheme_name,
    isin: h.isin ?? null,
    symbol: h.symbol ?? null,
    asset_type: h.asset_type,
    holding_mode: h.holding_mode ?? null,
    folio_number: h.folio_number ?? null,
    units: h.units,
    avg_cost_price: h.avg_cost_price ?? null,
    invested_value: h.invested_value ?? null,
    nav: h.nav ?? null,
    nav_date: h.nav_date ?? null,
    market_value: h.market_value ?? null,
    sector: h.sector ?? null,
    plan_type: h.plan_type ?? null,
    source: h.source,
    registrar: h.registrar ?? null,
  };
}

/** Hash folios with the salt, drop client ids, allowlist every holding. */
export async function finaliseHoldings(result: ClientParseResult, salt: string): Promise<ClientParseResult> {
  const holdings: PayloadHolding[] = [];
  for (const h of result.holdings) {
    const folio = h.folio_number ? await hashFolio(h.folio_number, salt) : null;
    holdings.push(toPayloadHolding({ ...h, folio_number: folio, client_id: null }));
  }
  return {
    source: result.source,
    filename: result.filename,
    holdings,
    warnings: [...result.warnings],
    statement_date: result.statement_date,
  };
}

/** Throws `PiiLeakError` if `json` matches a PII pattern or contains a redacted value. */
export function assertNoPii(json: string, redacted: readonly string[]): void {
  if (new RegExp(PAN_RE.source).test(json)) throw new PiiLeakError("pan");
  if (new RegExp(EMAIL_RE.source).test(json)) throw new PiiLeakError("email");
  if (new RegExp(IFSC_RE.source).test(json)) throw new PiiLeakError("ifsc");
  for (const value of redacted) {
    const v = value.trim();
    if (v.length < 4) continue;
    if (json.includes(v) || json.includes(JSON.stringify(v).slice(1, -1))) {
      throw new PiiLeakError("redacted-value");
    }
  }
}

/**
 * Build the exact `POST /api/analyze` body: one entry per source name, in
 * first-seen order, every holding through the allowlist, then the leak assertion.
 */
export function buildAnalyzePayload(
  sources: readonly { source: string; holdings: readonly (ClientHolding | PayloadHolding)[] }[],
  redacted: readonly string[] = [],
): AnalyzePayload {
  const bySource = new Map<string, PayloadHolding[]>();
  for (const entry of sources) {
    const list = bySource.get(entry.source) ?? [];
    for (const h of entry.holdings) list.push(toPayloadHolding(h));
    bySource.set(entry.source, list);
  }
  const payload: AnalyzePayload = {
    sources: Array.from(bySource, ([source, holdings]) => ({ source, holdings })),
  };
  assertNoPii(JSON.stringify(payload), redacted);
  return payload;
}
