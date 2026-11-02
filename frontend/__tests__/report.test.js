import { buildReportModel, buildHtmlReport, buildCsvReport, buildJsonReport, csvCell, escapeHtml } from "../src/lib/report";

const claim = (o = {}) => ({
  id: 1, text: "The Earth orbits the Sun.", sourceSentence: "The Earth orbits the Sun.",
  verdict: "Supported", confidence: 0.97,
  citations: [{ sourceId: "KB-001", text: "Earth orbits the Sun yearly.", relevance: 0.81 }], ...o,
});
const done = (claims, label = "Text 1") => ({ label, status: "done", result: { status: claims.length ? "ok" : "no_verifiable_claims", message: claims.length ? null : "No verifiable claims found", processingTimeMs: 812, claims } });
const when = new Date("2026-10-30T10:00:00Z");

const single = [done([claim(), claim({ id: 2, text: "The Sun orbits the Earth.", verdict: "Contradicted", confidence: 0.9 }), claim({ id: 3, text: "Bananas have potassium.", verdict: "Unverifiable", confidence: 0.7, citations: [] })])];
const batch = [done([claim()], "Item 1"), { label: "Item 2", status: "failed", error: "Server error" }, done([], "Item 3")];

test("model totals count verdicts, verified and not-verified items", () => {
  expect(buildReportModel(single, when).totals).toMatchObject({ items: 1, verified: 1, notVerified: 0, claims: 3, Supported: 1, Contradicted: 1, Unverifiable: 1 });
  expect(buildReportModel(batch, when).totals).toMatchObject({ items: 3, verified: 2, notVerified: 1, claims: 1 });
});

test("HTML report has claims, verdicts, confidence, evidence, processing time and disclaimer", () => {
  const html = buildHtmlReport(single, when);
  for (const s of ["The Earth orbits the Sun.", "Supported", "Contradicted", "Unverifiable", "97%", "KB-001", "Earth orbits the Sun yearly.", "relevance 81%", "812 ms", "does not eliminate", "2026-10-30T10:00:00.000Z", "No evidence found"]) {
    expect(html).toContain(s);
  }
});

test("HTML report escapes user/evidence text (no script injection)", () => {
  const html = buildHtmlReport([done([claim({ text: "<script>alert(1)</script>" })])], when);
  expect(html).not.toContain("<script>alert(1)</script>");
  expect(html).toContain("&lt;script&gt;");
  expect(escapeHtml(`a&b"c'`)).toBe("a&amp;b&quot;c&#39;");
});

test("batch report shows failed items and the no-claims message", () => {
  const html = buildHtmlReport(batch, when);
  expect(html).toContain("Not verified (failed): Server error");
  expect(html).toContain("No verifiable claims found");
  expect(html).toContain("Item 3");
});

test("CSV has a header, one row per claim, quoting and a row for failed items", () => {
  const csv = buildCsvReport(batch).replace("\uFEFF", "").split("\r\n");
  expect(csv[0]).toContain('"item","item_status","claim_id"');
  expect(csv).toHaveLength(4); // header + 1 claim + failed item + no-claims item
  expect(csv[1]).toContain('"KB-001"');
  expect(csv[2]).toContain('"Server error"');
});

test("CSV guards against spreadsheet formula injection and escapes quotes", () => {
  expect(csvCell("=HYPERLINK(\"x\")")).toBe(`"'=HYPERLINK(""x"")"`);
  expect(csvCell("+1")).toBe(`"'+1"`);
  expect(csvCell('say "hi"')).toBe('"say ""hi"""');
});

test("JSON report is valid and includes the disclaimer", () => {
  const parsed = JSON.parse(buildJsonReport(single, when));
  expect(parsed.totals.claims).toBe(3);
  expect(parsed.disclaimer).toMatch(/does not eliminate/);
  expect(parsed.entries[0].result.claims[0].verdict).toBe("Supported");
});
