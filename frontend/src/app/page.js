"use client";
import { useState, useEffect } from "react";
import NavBar from "../components/NavBar";
import Disclaimer from "../components/Disclaimer";
import ErrorBanner from "../components/ErrorBanner";
import Loader from "../components/Loader";
import ResultsView from "../components/ResultsView";
import BatchVerifier from "../components/BatchVerifier";
import ReportButtons from "../components/ReportButtons";
import { verifyText } from "../lib/api";
import { countWords, validateInput, MAX_WORDS } from "../lib/validate";

export default function Home() {
  const [text, setText] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("single"); // "single" | "batch"
  // Remember the API key in this browser so it survives page refreshes.
  useEffect(() => {
    try {
      setApiKey(localStorage.getItem("tl_api_key") || "");
    } catch {
      /* storage unavailable */
    }
  }, []);

  function handleKeyChange(e) {
    const value = e.target.value;
    setApiKey(value);
    try {
      localStorage.setItem("tl_api_key", value);
    } catch {
      /* storage unavailable */
    }
  }
  const words = countWords(text);

  async function handleVerify() {
    const problem = validateInput(text);
    if (problem) {
      setError(problem);
      setResult(null);
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      setResult(await verifyText(text, { apiKey }));
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <NavBar />
      <main className="container" id="verify">
        <h1>Verify AI-generated text</h1>

        <div className="tabs" role="tablist" aria-label="Verification mode">
          <button role="tab" aria-selected={mode === "single"} className={mode === "single" ? "tab active" : "tab"} onClick={() => setMode("single")}>
            Single text
          </button>
          <button role="tab" aria-selected={mode === "batch"} className={mode === "batch" ? "tab active" : "tab"} onClick={() => setMode("batch")}>
            Batch (up to 50)
          </button>
        </div>

        {mode === "single" && (
          <>
          <label htmlFor="input-text">Text to verify</label>
          <textarea
            id="input-text"
            rows={8}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste or type AI-generated text here…"
          />
          <div className="row">
            <span className={words > MAX_WORDS ? "count over" : "count"}>
              {words} / {MAX_WORDS} words
            </span>
            <button className="primary" onClick={handleVerify} disabled={loading}>
              {loading ? "Verifying…" : "Verify"}
            </button>
          </div>

          </>
        )}
        {/* Kept mounted (just hidden) so switching tabs doesn't lose a batch in progress. */}
        <div hidden={mode !== "batch"}>
          <BatchVerifier apiKey={apiKey} />
        </div>

        <details className="dev-settings">
          <summary>Developer settings</summary>
          <label htmlFor="api-key">API key</label>
          <input
            id="api-key"
            type="password"
            value={apiKey}
            onChange={handleKeyChange}
            placeholder="Only needed when the backend requires a key"
          />
        </details>

        {mode === "single" && (
          <>
            <ErrorBanner message={error} />
            {loading && <Loader />}
            <ResultsView result={result} />
            {result && <ReportButtons entries={[{ label: "Text 1", status: "done", result }]} />}
          </>
        )}
        <Disclaimer />
      </main>
    </>
  );
}
