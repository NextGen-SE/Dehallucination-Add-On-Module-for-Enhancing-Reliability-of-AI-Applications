# Sprint 2 – Suhani (frontend)

Run: `npm test` (47 tests). Batch input: separate items with a line containing only `---`.
- #19/#20: ClaimCard shows icon + verdict word, confidence bar, source sentence, citation relevance. `--muted` darkened for WCAG AA contrast.
- #22: `src/lib/batch.js` + `BatchVerifier.js` (client-side: one /verify call per item, 3 at a time, max 50, per-item status, 401 stops the rest).
- #23: `src/lib/report.js` + `ReportButtons.js` (HTML / CSV / JSON, single and batch). CSV cells starting with = + - @ are prefixed with ' to prevent spreadsheet formula injection.
