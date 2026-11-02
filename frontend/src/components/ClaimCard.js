"use client";
import { useState } from "react";

const CLASS_BY_VERDICT = {
  Supported: "supported",
  Contradicted: "contradicted",
  Unverifiable: "unverifiable",
};

// An icon + word accompany the colour so the verdict never depends on colour alone.
const ICON_BY_VERDICT = { Supported: "✓", Contradicted: "✗", Unverifiable: "?" };

export default function ClaimCard({ claim }) {
  const [open, setOpen] = useState(false);
  const cls = CLASS_BY_VERDICT[claim.verdict] || "unverifiable";
  const pct = Math.round(claim.confidence * 100);
  const hasCitations = claim.citations.length > 0;

  return (
    <li className={`claim-card ${cls}`} data-testid="claim-card">
      <div className="claim-head">
        <span className="claim-text">{claim.text}</span>
        <span className={`badge ${cls}`}>
          <span aria-hidden="true">{ICON_BY_VERDICT[claim.verdict] || "?"}</span> {claim.verdict}
        </span>
        <span className="confidence">{pct}% confidence</span>
      </div>
      <div className="confidence-bar" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
      {claim.sourceSentence && claim.sourceSentence !== claim.text && (
        <p className="source-sentence">From: “{claim.sourceSentence}”</p>
      )}

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
                  <small>
                    Source: {c.sourceId}
                    {c.relevance != null && ` · relevance ${Math.round(c.relevance * 100)}%`}
                  </small>
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
