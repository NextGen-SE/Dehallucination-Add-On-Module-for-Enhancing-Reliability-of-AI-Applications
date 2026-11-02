"use client";
import { buildHtmlReport, buildCsvReport, buildJsonReport, downloadFile, fileStamp } from "../lib/report";

// entries: [{ label, status, result?, error? }]
export default function ReportButtons({ entries }) {
  if (!entries || entries.length === 0) return null;

  function save(kind) {
    const stamp = fileStamp();
    if (kind === "html") downloadFile(`truthlayer-report-${stamp}.html`, buildHtmlReport(entries), "text/html;charset=utf-8");
    if (kind === "csv") downloadFile(`truthlayer-report-${stamp}.csv`, buildCsvReport(entries), "text/csv;charset=utf-8");
    if (kind === "json") downloadFile(`truthlayer-report-${stamp}.json`, buildJsonReport(entries), "application/json");
  }

  return (
    <div className="report-buttons" role="group" aria-label="Download report">
      <span className="muted-label">Download report:</span>
      <button className="secondary" onClick={() => save("html")}>HTML</button>
      <button className="secondary" onClick={() => save("csv")}>CSV</button>
      <button className="secondary" onClick={() => save("json")}>JSON</button>
    </div>
  );
}
