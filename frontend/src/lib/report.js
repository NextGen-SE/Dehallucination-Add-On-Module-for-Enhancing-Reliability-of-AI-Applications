// Downloadable verification reports (Issue #23).
// Built entirely in the browser from results the page already holds.
// An "entry" is one verified text: { label, status, result?, error? }
// where status is done | failed | invalid | skipped.

const VERDICTS = ["Supported", "Contradicted", "Unverifiable"];
const ICON = { Supported: "✓", Contradicted: "✗", Unverifiable: "?" };
const COLOR = { Supported: "#1b7f3b", Contradicted: "#b3261e", Unverifiable: "#9a6700" };

export const REPORT_DISCLAIMER =
  "TruthLayer reduces, but does not eliminate, the risk of hallucinated content. " +
  "It is not a substitute for human review in high-stakes use cases.";

export function buildReportModel(entries, generatedAt = new Date()) {
  const totals = { items: entries.length, verified: 0, notVerified: 0, claims: 0, Supported: 0, Contradicted: 0, Unverifiable: 0 };
  for (const e of entries) {
    if (e.status === "done" && e.result) {
      totals.verified += 1;
      for (const c of e.result.claims) {
        totals.claims += 1;
        if (VERDICTS.includes(c.verdict)) totals[c.verdict] += 1;
      }
    } else {
      totals.notVerified += 1;
    }
  }
  return { generatedAt: generatedAt.toISOString(), totals, entries };
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function pct(x) {
  return `${Math.round((x ?? 0) * 100)}%`;
}

function claimHtml(c) {
  const color = COLOR[c.verdict] || COLOR.Unverifiable;
  const cites = c.citations.length
    ? `<ul>${c.citations
        .map(
          (ct) =>
            `<li><blockquote>${escapeHtml(ct.text)}</blockquote><small>Source: ${escapeHtml(ct.sourceId)}${
              ct.relevance != null ? ` · relevance ${pct(ct.relevance)}` : ""
            }</small></li>`
        )
        .join("")}</ul>`
    : `<p class="muted">No evidence found in the knowledge base.</p>`;
  const from =
    c.sourceSentence && c.sourceSentence !== c.text
      ? `<p class="muted">From: “${escapeHtml(c.sourceSentence)}”</p>`
      : "";
  return `<li class="claim" style="border-left-color:${color}">
<p><strong>${escapeHtml(c.text)}</strong></p>
<p><span class="badge" style="background:${color}">${ICON[c.verdict] || "?"} ${escapeHtml(c.verdict)}</span> Confidence: ${pct(c.confidence)}</p>
${from}${cites}</li>`;
}

function entryHtml(e) {
  const head = `<h2>${escapeHtml(e.label)}</h2>`;
  if (e.status !== "done" || !e.result) {
    return `<section>${head}<p class="problem">Not verified (${escapeHtml(e.status)}): ${escapeHtml(e.error || "no result")}</p></section>`;
  }
  const r = e.result;
  const meta = `<p class="muted">Status: ${escapeHtml(r.status)}${
    r.processingTimeMs != null ? ` · processing time: ${r.processingTimeMs} ms` : ""
  }</p>`;
  if (r.claims.length === 0) {
    return `<section>${head}${meta}<p>${escapeHtml(r.message || "No verifiable claims found")}</p></section>`;
  }
  return `<section>${head}${meta}<ul class="claims">${r.claims.map(claimHtml).join("")}</ul></section>`;
}

export function buildHtmlReport(entries, generatedAt = new Date()) {
  const m = buildReportModel(entries, generatedAt);
  const t = m.totals;
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>TruthLayer verification report</title>
<style>
body{font-family:system-ui,sans-serif;color:#1f2328;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.45}
.muted{color:#57606a;font-size:.9rem}.problem{color:#b3261e;font-weight:600}
.claims{list-style:none;padding:0}.claim{border:1px solid #d0d7de;border-left:6px solid;border-radius:6px;padding:8px 14px;margin:10px 0;page-break-inside:avoid}
.badge{color:#fff;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600}
blockquote{margin:4px 0;padding-left:10px;border-left:3px solid #d0d7de}
.disclaimer{margin-top:28px;border-top:1px solid #d0d7de;padding-top:10px;font-size:.8rem;color:#57606a}
</style></head><body>
<h1>TruthLayer verification report</h1>
<p class="muted">Generated: ${escapeHtml(m.generatedAt)}</p>
<p><strong>${t.items}</strong> text(s), <strong>${t.verified}</strong> verified, <strong>${t.notVerified}</strong> not verified.
<strong>${t.claims}</strong> claims: ${t.Supported} supported, ${t.Contradicted} contradicted, ${t.Unverifiable} unverifiable.</p>
${entries.map(entryHtml).join("\n")}
<p class="disclaimer">${escapeHtml(REPORT_DISCLAIMER)}</p>
</body></html>`;
}

export function buildJsonReport(entries, generatedAt = new Date()) {
  return JSON.stringify({ ...buildReportModel(entries, generatedAt), disclaimer: REPORT_DISCLAIMER }, null, 2);
}

// Spreadsheet apps run text starting with = + - @ as formulas, so prefix those.
export function csvCell(value) {
  let s = String(value ?? "");
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  return `"${s.replace(/"/g, '""')}"`;
}

const CSV_HEADER = [
  "item", "item_status", "claim_id", "claim", "source_sentence", "verdict", "confidence",
  "evidence_source_ids", "evidence_text", "processing_time_ms", "error",
];

export function buildCsvReport(entries) {
  const rows = [CSV_HEADER];
  for (const e of entries) {
    const r = e.status === "done" ? e.result : null;
    if (!r || r.claims.length === 0) {
      rows.push([e.label, r ? r.status : e.status, "", "", "", "", "", "", "", r?.processingTimeMs ?? "", e.error || r?.message || ""]);
      continue;
    }
    for (const c of r.claims) {
      rows.push([
        e.label, r.status, c.id, c.text, c.sourceSentence ?? "", c.verdict, c.confidence,
        c.citations.map((x) => x.sourceId).join("; "),
        c.citations.map((x) => x.text).join(" | "),
        r.processingTimeMs ?? "", "",
      ]);
    }
  }
  return "\uFEFF" + rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
}

export function fileStamp(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

export function downloadFile(filename, content, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
