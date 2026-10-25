import { mockVerify } from "./mock";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// Converts whatever the backend sends into ONE shape the UI uses.
// Accepts both the SRS names (text, citations) and the Person-2 module names
// (claim_text, evidence), so a naming mismatch cannot break the UI.
export function normalizeResponse(data) {
  const rawClaims = data.claims || [];
  return {
    status: data.status || (rawClaims.length ? "ok" : "no_verifiable_claims"),
    message: data.message || null,
    processingTimeMs: data.processing_time_ms ?? null,
    claims: rawClaims.map((c, i) => ({
      id: c.claim_id ?? i + 1,
      text: c.text ?? c.claim_text ?? "",
      sourceSentence: c.source_sentence ?? null,
      verdict: c.verdict,
      confidence: c.confidence ?? 0,
      citations: (c.citations ?? c.evidence ?? []).map((e) => ({
        sourceId: e.source_id,
        text: e.text ?? e.passage_text ?? "",
        relevance: e.relevance_score ?? null,
      })),
    })),
  };
}

const STATUS_MESSAGES = {
  400: "The server rejected the input (malformed or too long).",
  401: "Missing or invalid API key. Open Developer settings and check your key.",
  500: "The verification pipeline failed on the server. Please try again.",
};

export async function verifyText(text, { apiKey, knowledgeBaseId, signal } = {}) {
  if (USE_MOCK) return normalizeResponse(await mockVerify(text));

  let res;
  try {
    res = await fetch(`${API_URL}/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
      },
      body: JSON.stringify({
        text,
        ...(knowledgeBaseId ? { knowledge_base_id: knowledgeBaseId } : {}),
      }),
      signal,
    });
  } catch {
    throw new ApiError("Cannot reach the TruthLayer server. Check that the API is running.", 0);
  }

  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : body.message || "";
    } catch {
      /* body was not JSON */
    }
    const base = STATUS_MESSAGES[res.status] || `Unexpected server error (${res.status}).`;
    throw new ApiError(detail ? `${base} ${detail}` : base, res.status);
  }
  return normalizeResponse(await res.json());
}
