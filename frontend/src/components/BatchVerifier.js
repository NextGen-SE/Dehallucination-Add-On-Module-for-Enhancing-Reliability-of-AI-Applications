"use client";
import { useState } from "react";
import ResultsView from "./ResultsView";
import ReportButtons from "./ReportButtons";
import ErrorBanner from "./ErrorBanner";
import { verifyText } from "../lib/api";
import { MAX_BATCH_ITEMS, parseBatchInput, validateBatch, runBatch } from "../lib/batch";

const STATUS_LABEL = {
  pending: "Waiting",
  running: "Verifying…",
  done: "Done",
  failed: "Failed",
  invalid: "Invalid",
  skipped: "Skipped",
};

function preview(text) {
  return text.length > 90 ? `${text.slice(0, 90)}…` : text;
}

export default function BatchVerifier({ apiKey }) {
  const [raw, setRaw] = useState("");
  const [rows, setRows] = useState([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const itemCount = parseBatchInput(raw).length;
  const finished = rows.filter((r) => !["pending", "running"].includes(r.status)).length;

  async function handleRun() {
    const items = parseBatchInput(raw);
    const problem = validateBatch(items);
    if (problem) {
      setError(problem);
      setRows([]);
      return;
    }
    setError(null);
    setRows(items.map((it) => ({ ...it, status: "pending" })));
    setRunning(true);
    try {
      await runBatch(items, (text) => verifyText(text, { apiKey }), {
        onUpdate: (i, patch) =>
          setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r))),
      });
    } finally {
      setRunning(false);
    }
  }

  const count = (s) => rows.filter((r) => r.status === s).length;
  const entries = rows
    .filter((r) => !["pending", "running"].includes(r.status))
    .map((r) => ({ label: `Item ${r.id}`, status: r.status, result: r.result, error: r.error }));

  return (
    <section aria-label="Batch verification">
      <label htmlFor="batch-text">Texts to verify (separate items with a line containing only ---)</label>
      <textarea
        id="batch-text"
        rows={10}
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        placeholder={"First AI-generated text…\n---\nSecond AI-generated text…"}
      />
      <div className="row">
        <span className={itemCount > MAX_BATCH_ITEMS ? "count over" : "count"}>
          {itemCount} / {MAX_BATCH_ITEMS} items
        </span>
        <button className="primary" onClick={handleRun} disabled={running}>
          {running ? "Verifying batch…" : "Verify batch"}
        </button>
      </div>

      <ErrorBanner message={error} />

      {rows.length > 0 && (
        <>
          <p className="summary" role="status">
            {finished} of {rows.length} finished · {count("done")} done · {count("failed")} failed ·{" "}
            {count("invalid")} invalid · {count("skipped")} skipped
          </p>
          <progress className="batch-progress" value={finished} max={rows.length} aria-label="Batch progress" />
          {!running && <ReportButtons entries={entries} />}
          <ul className="batch-list">
            {rows.map((r) => (
              <li key={r.id} className={`batch-item ${r.status}`} data-testid="batch-item">
                <div className="batch-head">
                  <strong>Item {r.id}</strong>
                  <span className={`status-pill ${r.status}`}>{STATUS_LABEL[r.status]}</span>
                  <span className="muted-label">{r.words} words</span>
                </div>
                <p className="batch-preview">{preview(r.text)}</p>
                {r.error && <p className="item-error">{r.error}</p>}
                {r.status === "done" && (
                  <details>
                    <summary>
                      {r.result.claims.length} claim{r.result.claims.length === 1 ? "" : "s"} – view results
                    </summary>
                    <ResultsView result={r.result} />
                  </details>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
