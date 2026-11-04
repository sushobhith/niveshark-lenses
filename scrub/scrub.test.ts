/**
 * Statement scrubber tests (D-026). All data is invented.
 *
 * Statements are built as `ExtractedDocument` rows directly, shaped like
 * pdf.js output for the real MFCentral and CAMS summaries, and run through
 * the same detect → scrub → parse → finalise → payload path the browser uses.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { assertNoPii, buildAnalyzePayload, finaliseHoldings, hashFolio, scrubRows } from "./scrub.ts";
import { parseExtractedDocument, redactedValuesFor } from "./test-support.ts";
import { PiiLeakError } from "./types.ts";
import type { ClientParseResult, ExtractedDocument, ExtractedPage, TextRow } from "./types.ts";

const PAN_RE = /[A-Z]{5}[0-9]{4}[A-Z]/;

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

function row(y: number, ...cells: [number, string][]): TextRow {
  const sorted = cells.map(([x, text]) => ({ x, width: text.length * 4, text })).sort((a, b) => a.x - b.x);
  return { y, cells: sorted, text: sorted.map((c) => c.text).join(" ") };
}

function page(n: number, rows: TextRow[]): ExtractedPage {
  return { page: n, rows: [...rows].sort((a, b) => b.y - a.y) };
}

interface Pii {
  pan: string;
  name: string;
  address: string;
  email: string;
  phone: string;
  bankAccount: string;
  ifsc: string;
  nominee: string;
  kyc: string;
  dob: string;
}

interface SynthHolding {
  folio: string;
  isin: string;
  scheme: string;
  wrap: string;
  units: string;
  nav: string;
  value: string;
}

const FIXED_PII: Pii = {
  pan: "ABCDE1234F",
  name: "ASHA TESTER",
  address: "12 Test Lane, Sample Nagar, Pune 411001",
  email: "asha.tester@example.com",
  phone: "9876501234",
  bankAccount: "001234567890",
  ifsc: "EXMP0123456",
  nominee: "RAVI TESTER",
  kyc: "KRA-VALIDATED",
  dob: "01-Jan-1990",
};

const FIXED_HOLDINGS: SynthHolding[] = [
  { folio: "91234567", isin: "INF200K01234", scheme: "Sample Bluechip Fund -", wrap: "Direct Growth", units: "1,000.000", nav: "56.3900", value: "56,390.00" },
  { folio: "88776655/12", isin: "INF109K01Z98", scheme: "Sample Flexi Cap Fund -", wrap: "Regular Growth", units: "3,210.000", nav: "52.1000", value: "1,67,241.00" },
];

/** MFCentral-shaped: investor block on page 1, SoA table on page 2. */
function mfcentralDoc(pii: Pii, holdings: SynthHolding[], opts: { plantInTable?: boolean } = {}): ExtractedDocument {
  const title = [row(546, [306, "Consolidated Account Summary"]), row(529, [348, "( As on Date: 10-Mar-2026 )"])];
  const p1 = page(1, [
    ...title,
    row(491, [43, `PAN : ${pii.pan}`], [383, "The Consolidated Account Statement is brought to you by MFCentral"]),
    row(478, [43, pii.name], [383, "a platform of CAMS and KFintech"]),
    row(465, [43, `Email : ${pii.email}`]),
    row(452, [43, `Mobile : ${pii.phone}`]),
    row(439, [43, pii.address]),
    row(426, [43, `Nominee : ${pii.nominee}`]),
    row(413, [43, `Bank A/c ${pii.bankAccount} IFSC ${pii.ifsc}`]),
    row(400, [43, `Date of Birth : ${pii.dob}`]),
    row(387, [43, `KYC : ${pii.kyc}`]),
    // A two-column preamble row where an "as on" sentence shares a y with the name.
    row(374, [43, pii.name], [383, "Valuation is as on the statement date"]),
  ]);
  const tableRows: TextRow[] = [
    ...title,
    row(458, [332, "Invested Value"], [436, "Balance"], [660, "Market Value"], [753, "Gain/Loss"]),
    row(452, [43, "Folio No."], [171, "Scheme Details"], [508, "NAV Date"], [595, "NAV"]),
    row(445, [354, "(INR)"], [443, "Units"], [678, "(INR)"]),
  ];
  let y = 418;
  for (const h of holdings) {
    tableRows.push(
      row(y, [34, h.folio], [102, h.scheme], [372, "10,000.00"], [454, h.units], [508, "09-Mar-2026"], [610, h.nav], [693, h.value]),
      row(y - 8, [102, h.wrap]),
    );
    y -= 33;
  }
  if (opts.plantInTable) {
    // PII inside the table region: value patterns and labelled rows.
    tableRows.push(
      row(y + 20, [102, `Holder ${pii.pan} ${pii.email}`]),
      row(y + 12, [43, `Nominee: ${pii.nominee}`]),
      row(y + 4, [43, `Mobile ${pii.phone}`]),
      row(y - 4, [43, `Bank A/c ${pii.bankAccount}`], [300, `IFSC ${pii.ifsc}`]),
      row(y - 12, [43, `KYC Status: ${pii.kyc}`]),
      row(y - 20, [43, "Holder"], [102, "DOB"], [140, pii.dob]),
    );
  }
  tableRows.push(row(y - 40, [593, "Total"], [684, "2,23,631.00"]));
  return { pages: [p1, page(2, tableRows)] };
}

/** CAMS-shaped: investor block above the table on page 1, rows anchored on the ISIN. */
function camsDoc(pii: Pii, holdings: SynthHolding[], opts: { plantInTable?: boolean } = {}): ExtractedDocument {
  const rows: TextRow[] = [
    row(783, [168, "Consolidated Account Summary"]),
    row(768, [248, "As on 10-Mar-2026"]),
    row(738, [35, pii.name], [306, "This Account Summary is brought to you by CAMS and KFintech"]),
    row(720, [35, pii.address], [306, "Values are as on the date above"]),
    row(697, [35, `Email Id: ${pii.email}`]),
    row(687, [35, `Mobile: ${pii.phone}`]),
    row(677, [35, `PAN: ${pii.pan}`]),
    row(667, [35, `Nominee: ${pii.nominee}`]),
    row(658, [35, `Bank A/c ${pii.bankAccount} IFSC ${pii.ifsc}`]),
    row(648, [35, `DOB: ${pii.dob}`]),
    row(640, [35, `KYC: ${pii.kyc}`]),
    row(604, [29, "Folio No."], [76, "ISIN"], [125, "Scheme Name"], [287, "Cost Value"], [341, "Unit Balance"], [393, "NAV Date"], [451, "NAV"], [490, "Market Value"], [538, "Registrar"]),
    row(597, [305, "(INR)"], [449, "(INR)"], [516, "(INR)"]),
  ];
  let y = 586;
  for (const h of holdings) {
    rows.push(
      row(y, [30, h.folio], [76, h.isin], [125, `L686G - ${h.scheme}`], [285, "10,000.000"], [349, h.units], [384, "09-Mar-2026"], [442, h.nav], [498, h.value], [551, "CAMS"]),
      row(y - 9, [125, `${h.wrap} (Non-Demat)`]),
    );
    y -= 18;
  }
  if (opts.plantInTable) {
    rows.push(
      row(y, [125, `${pii.name} ${pii.pan}`]),
      row(y - 9, [29, `Email: ${pii.email}`]),
      row(y - 30, [29, `IFSC ${pii.ifsc}`], [200, "branch code"]),
    );
  }
  rows.push(row(y - 60, [203, "Total"], [285, "20,000.00"], [495, "2,23,631.00"]));
  return { pages: [page(1, rows)] };
}

async function pipeline(doc: ExtractedDocument, filename: string) {
  const result = await parseExtractedDocument(doc, filename);
  const redacted = redactedValuesFor(result);
  const payload = buildAnalyzePayload([result], redacted);
  return { result, json: JSON.stringify(payload) };
}

function scrubbedJson(doc: ExtractedDocument, source: "mfcentral" | "cams"): string {
  return JSON.stringify(scrubRows(doc, source).doc);
}

// ---------------------------------------------------------------------------
// SB-1 .. SB-9: every PII kind absent from the scrubbed rows and the payload
// ---------------------------------------------------------------------------

const CASES = [
  { label: "MFCentral", filename: "MFCentral-CAS.pdf", source: "mfcentral" as const, build: mfcentralDoc },
  { label: "CAMS", filename: "CAMS-CAS.pdf", source: "cams" as const, build: camsDoc },
];

const KINDS: { id: string; kind: string; values: (p: Pii) => string[] }[] = [
  { id: "SB-1", kind: "PAN", values: (p) => [p.pan] },
  { id: "SB-2", kind: "investor name", values: (p) => [p.name] },
  { id: "SB-3", kind: "address", values: (p) => [p.address, "Sample Nagar"] },
  { id: "SB-4", kind: "email", values: (p) => [p.email] },
  { id: "SB-5", kind: "phone", values: (p) => [p.phone] },
  { id: "SB-6", kind: "bank account and IFSC", values: (p) => [p.bankAccount, p.ifsc] },
  { id: "SB-7", kind: "nominee", values: (p) => [p.nominee] },
  { id: "SB-8", kind: "KYC status", values: (p) => [p.kyc] },
  { id: "SB-9", kind: "date of birth", values: (p) => [p.dob] },
];

for (const kind of KINDS) {
  test(`${kind.id} ${kind.kind} is absent from scrubbed rows and payload`, async () => {
    for (const c of CASES) {
      for (const plantInTable of [false, true]) {
        // CAMS planting puts the name inside a scheme cell, which only the
        // allowlist can keep out of the payload; check the rows only for PAN/email/IFSC there.
        const doc = c.build(FIXED_PII, FIXED_HOLDINGS, { plantInTable });
        const rows = scrubbedJson(doc, c.source);
        const { result, json } = await pipeline(doc, c.filename);
        assert.equal(result.holdings.length, FIXED_HOLDINGS.length, `${c.label}: holdings kept`);
        assert.doesNotMatch(json, PAN_RE, `${c.label}: PAN pattern in payload`);
        for (const v of kind.values(FIXED_PII)) {
          const nameInCamsScheme = c.source === "cams" && plantInTable && kind.kind === "investor name";
          if (!nameInCamsScheme) {
            assert.ok(!rows.includes(v), `${c.label}${plantInTable ? " (planted)" : ""}: ${kind.kind} in scrubbed rows`);
          }
          if (!nameInCamsScheme) {
            assert.ok(!json.includes(v), `${c.label}${plantInTable ? " (planted)" : ""}: ${kind.kind} in payload`);
          }
        }
      }
    }
  });
}

test("SB-9b statement date survives the preamble drop; the name sharing its row does not", () => {
  const out = scrubRows(mfcentralDoc(FIXED_PII, FIXED_HOLDINGS), "mfcentral").doc;
  const p1 = out.pages[0].rows.map((r) => r.text);
  assert.ok(p1.some((t) => t.includes("As on Date: 10-Mar-2026")));
  assert.ok(p1.some((t) => t === "Valuation is as on the statement date"));
  assert.ok(!JSON.stringify(out).includes(FIXED_PII.name));
});

// ---------------------------------------------------------------------------
// SB-10: folio hashing
// ---------------------------------------------------------------------------

test("SB-10 folio hash is 16 lowercase hex, stable within a salt, different across salts", async () => {
  const a1 = await hashFolio("91234567", "salt-a");
  const a2 = await hashFolio("91234567", "salt-a");
  const b = await hashFolio("91234567", "salt-b");
  const other = await hashFolio("91234568", "salt-a");
  assert.match(a1, /^[0-9a-f]{16}$/);
  assert.equal(a1, a2);
  assert.notEqual(a1, b);
  assert.notEqual(a1, other);

  const result: ClientParseResult = {
    source: "mfcentral",
    filename: "x.pdf",
    statement_date: null,
    warnings: [],
    holdings: [
      {
        scheme_name: "Sample Fund", isin: null, symbol: null, asset_type: "mutual_fund", holding_mode: "soa",
        folio_number: "91234567", units: 1, avg_cost_price: null, invested_value: null, nav: null, nav_date: null,
        market_value: null, sector: null, plan_type: null, source: "mfcentral", registrar: null,
      },
      {
        scheme_name: "Sample ETF", isin: null, symbol: null, asset_type: "etf", holding_mode: "demat",
        folio_number: null, client_id: "1208160012345678", units: 1, avg_cost_price: null, invested_value: null,
        nav: null, nav_date: null, market_value: null, sector: null, plan_type: null, source: "mfcentral", registrar: null,
      },
    ],
  };
  const fa = await finaliseHoldings(result, "salt-a");
  const fb = await finaliseHoldings(result, "salt-b");
  assert.equal(fa.holdings[0].folio_number, a1);
  assert.equal(fb.holdings[0].folio_number, b);
  assert.equal(fa.holdings[1].folio_number, null);
  assert.equal("client_id" in fa.holdings[1], false);
  const json = JSON.stringify(buildAnalyzePayload([fa]));
  assert.ok(!json.includes("91234567"));
  assert.ok(!json.includes("1208160012345678"));

  // Through the full pipeline: every folio in the payload is a 16-hex hash.
  const { json: pj } = await pipeline(mfcentralDoc(FIXED_PII, FIXED_HOLDINGS), "MFCentral-CAS.pdf");
  const folios = (JSON.parse(pj).sources[0].holdings as { folio_number: string }[]).map((h) => h.folio_number);
  assert.equal(folios.length, 2);
  for (const f of folios) assert.match(f, /^[0-9a-f]{16}$/);
  for (const h of FIXED_HOLDINGS) assert.ok(!pj.includes(h.folio));
});

// ---------------------------------------------------------------------------
// SB-11: property test with a seeded PRNG
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomPii(rand: () => number): Pii {
  const pick = (chars: string, n: number) => Array.from({ length: n }, () => chars[Math.floor(rand() * chars.length)]).join("");
  const UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const LO = "abcdefghijklmnopqrstuvwxyz";
  const DIG = "0123456789";
  const word = () => pick(UP, 5 + Math.floor(rand() * 4));
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return {
    pan: pick(UP, 5) + pick(DIG, 4) + pick(UP, 1),
    name: `${word()} ${word()}`,
    address: `${1 + Math.floor(rand() * 400)} ${word()} Road, ${word()} Nagar`,
    email: `${pick(LO, 6)}.${pick(LO, 5)}@${pick(LO, 7)}.com`,
    phone: pick("6789", 1) + pick(DIG, 9),
    bankAccount: pick(DIG, 9 + Math.floor(rand() * 8)),
    ifsc: pick(UP, 4) + "0" + pick(UP + DIG, 6),
    nominee: `${word()} ${word()}`,
    kyc: `KYC-${word()}`,
    dob: `${String(1 + Math.floor(rand() * 28)).padStart(2, "0")}-${months[Math.floor(rand() * 12)]}-${1950 + Math.floor(rand() * 50)}`,
  };
}

const SCHEME_WORDS = ["Sample", "Bluechip", "Flexi", "Cap", "Value", "Index", "Nifty", "Midcap", "Debt", "Liquid", "Hybrid", "Fund"];

function randomHoldings(rand: () => number): SynthHolding[] {
  const n = 1 + Math.floor(rand() * 4);
  return Array.from({ length: n }, (_, i) => {
    const words = Array.from({ length: 2 + Math.floor(rand() * 3) }, () => SCHEME_WORDS[Math.floor(rand() * SCHEME_WORDS.length)]);
    const units = (1 + rand() * 9999).toFixed(3);
    const nav = (1 + rand() * 999).toFixed(4);
    return {
      folio: String(10_000_000 + Math.floor(rand() * 89_999_999)) + (rand() < 0.3 ? "/" + (1 + Math.floor(rand() * 99)) : ""),
      isin: `INF${String(100 + i)}K01${String(Math.floor(rand() * 1000)).padStart(3, "0")}`,
      scheme: words.join(" ") + " -",
      wrap: rand() < 0.5 ? "Direct Growth" : "Regular Growth",
      units,
      nav,
      value: (Number(units) * Number(nav)).toFixed(2),
    };
  });
}

test("SB-11 property: 300 random MFCentral/CAMS statements never leak planted PII", async () => {
  const rand = mulberry32(0x5eed2026);
  for (let i = 0; i < 300; i++) {
    const pii = randomPii(rand);
    const holdings = randomHoldings(rand);
    const isCams = rand() < 0.5;
    const plantInTable = rand() < 0.6;
    const doc = isCams ? camsDoc(pii, holdings, { plantInTable }) : mfcentralDoc(pii, holdings, { plantInTable });
    const { result, json } = await pipeline(doc, isCams ? "CAMS.pdf" : "MFCentral.pdf");
    const ctx = `iteration ${i} (${isCams ? "cams" : "mfcentral"}${plantInTable ? ", planted" : ""})`;
    assert.equal(result.holdings.length, holdings.length, `${ctx}: holdings kept`);
    assert.doesNotMatch(json, PAN_RE, `${ctx}: PAN pattern in payload`);
    const planted = [pii.pan, pii.email, pii.phone, pii.bankAccount, pii.ifsc, pii.nominee, pii.kyc, pii.dob, pii.address];
    // The name is planted inside a CAMS scheme cell in the planted variant (the case D-026 cites);
    // outside that case it must be gone too.
    if (!(isCams && plantInTable)) planted.push(pii.name);
    for (const v of planted) assert.ok(!json.includes(v), `${ctx}: planted value in payload`);
    for (const h of holdings) assert.ok(!json.includes(`"${h.folio}"`), `${ctx}: raw folio in payload`);
  }
});

// ---------------------------------------------------------------------------
// SB-12: the leak assertion itself
// ---------------------------------------------------------------------------

test("SB-12 assertNoPii throws on PAN, email, IFSC and redacted values", () => {
  assert.throws(() => assertNoPii(`{"scheme_name":"x ${FIXED_PII.pan}"}`, []), PiiLeakError);
  assert.throws(() => assertNoPii(`{"scheme_name":"${FIXED_PII.email}"}`, []), PiiLeakError);
  assert.throws(() => assertNoPii(`{"scheme_name":"${FIXED_PII.ifsc}"}`, []), PiiLeakError);
  assert.throws(() => assertNoPii(`{"scheme_name":"ASHA TESTER Fund"}`, ["ASHA TESTER"]), PiiLeakError);
  assert.doesNotThrow(() => assertNoPii(`{"scheme_name":"Sample Fund","isin":"INF200K01234"}`, ["ASHA TESTER"]));
  try {
    assertNoPii(`{"x":"${FIXED_PII.pan}"}`, []);
  } catch (err) {
    assert.ok(!(err as Error).message.includes(FIXED_PII.pan), "error message never carries the value");
  }
});

test("SB-13 a wrapped scheme line starting with an ambiguous label word survives; a labelled account row does not", () => {
  const cell = (x: number, text: string) => ({ x, width: 50, text });
  const mk = (y: number, ...cells: { x: number; width: number; text: string }[]) => ({
    y,
    cells,
    text: cells.map((c) => c.text).join(" "),
  });
  const doc: ExtractedDocument = {
    pages: [
      {
        page: 1,
        rows: [
          mk(700, cell(35, "Folio No."), cell(90, "ISIN"), cell(160, "Scheme Name"), cell(400, "Unit Balance")),
          mk(690, cell(35, "12345678"), cell(90, "INF204KB14I2"), cell(160, "NPBEES - Nippon India ETF"), cell(400, "10.000")),
          mk(681, cell(160, "Bank BeES")),
          mk(670, cell(35, "Bank A/c 001234567890")),
          mk(660, cell(35, "Nominee: RAVI TESTER")),
        ],
      },
    ],
  };
  const { doc: out } = scrubRows(doc, "cams");
  const texts = out.pages[0].rows.map((r) => r.text);
  assert.ok(texts.includes("Bank BeES"), "wrapped 'Bank BeES' line must survive");
  assert.ok(!texts.some((t) => t.includes("001234567890")), "bank account row must be dropped");
  assert.ok(!texts.some((t) => t.includes("RAVI TESTER")), "nominee row must be dropped");
});
