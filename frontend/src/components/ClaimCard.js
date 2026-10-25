"use client";
import { useState } from "react";

const CLASS_BY_VERDICT = {
  Supported: "supported",
  Contradicted: "contradicted",
  Unverifiable: "unverifiable",
};

export default function ClaimCard({ claim }) {
  const [open, setOpen] = useState(false);
  const cls = CLASS_BY_VERDICT[claim.verdict] || "unverifiable";
  const pct = Math.round(claim.confidence * 100);
  const hasCitations = claim.citations.length > 0;

  return (
    <li className={`claim-card ${cls}`} data-testid="claim-card">
      <div className="claim-head">
        <span className="claim-text">{claim.text}</span>
        <span className={`badge ${cls}`}>{claim.verdict}</span>
        <span className="confidence">{pct}% confidence</span>
      </div>

      {hasCitations ? (
        <>
          <button className="link-btn" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide evidence" : "Show evidence"} ({claim.citations.length})
          </button>
          {open && (
            <ul className="citations">
              {claim.citations.map((c, i) => (
                <li key={`${c.sourceId}-${i}`}>
                  <blockquote>{c.text}</blockquote>
                  <small>Source: {c.sourceId}</small>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <p className="no-evidence">No supporting evidence found in the knowledge base.</p>
      )}
    </li>
  );
}
