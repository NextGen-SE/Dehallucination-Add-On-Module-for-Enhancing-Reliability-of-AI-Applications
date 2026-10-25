// Fake backend so the frontend can be built and demoed before the real API is ready.
const DEMO = {
  status: "ok",
  processing_time_ms: 812,
  claims: [
    {
      claim_id: 1,
      text: "The Earth orbits the Sun.",
      source_sentence: "The Earth orbits the Sun.",
      verdict: "Supported",
      confidence: 0.97,
      citations: [
        { source_id: "KB-001", text: "The Earth orbits the Sun once approximately every 365.25 days.", relevance_score: 0.81 },
      ],
    },
    {
      claim_id: 2,
      text: "The Sun orbits the Earth.",
      source_sentence: "The Sun orbits the Earth.",
      verdict: "Contradicted",
      confidence: 0.91,
      citations: [
        { source_id: "KB-001", text: "The Earth orbits the Sun once approximately every 365.25 days.", relevance_score: 0.78 },
      ],
    },
    {
      claim_id: 3,
      text: "Bananas are a good source of potassium.",
      source_sentence: "Bananas are a good source of potassium.",
      verdict: "Unverifiable",
      confidence: 0.74,
      citations: [],
    },
  ],
};

export async function mockVerify(text) {
  await new Promise((r) => setTimeout(r, 800));
  const looksFactual = /\b(is|are|was|were|has|have|orbits|contains)\b/i.test(text);
  if (!looksFactual) {
    return { status: "no_verifiable_claims", message: "No verifiable claims found", claims: [] };
  }
  return DEMO;
}
