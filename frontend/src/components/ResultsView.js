import ClaimCard from "./ClaimCard";

export default function ResultsView({ result }) {
  if (!result) return null;

  if (result.status === "no_verifiable_claims" || result.claims.length === 0) {
    return <p className="empty-state">{result.message || "No verifiable claims found"}</p>;
  }

  const count = (v) => result.claims.filter((c) => c.verdict === v).length;

  return (
    <section aria-label="Verification results">
      <p className="summary">
        {result.claims.length} claims: {count("Supported")} supported, {count("Contradicted")}{" "}
        contradicted, {count("Unverifiable")} unverifiable
        {result.processingTimeMs != null && ` · ${result.processingTimeMs} ms`}
      </p>
      <ul className="claim-list">
        {result.claims.map((c) => (
          <ClaimCard key={c.id} claim={c} />
        ))}
      </ul>
    </section>
  );
}
