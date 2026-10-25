import { normalizeResponse, verifyText, ApiError } from "../src/lib/api";

const okBody = {
  processing_time_ms: 120,
  claims: [
    {
      claim_id: 1,
      claim_text: "The Earth orbits the Sun.",
      verdict: "Supported",
      confidence: 0.95,
      evidence: [{ source_id: "KB-001", text: "Earth orbits the Sun.", relevance_score: 0.8 }],
    },
  ],
};

beforeEach(() => {
  global.fetch = jest.fn(); // jsdom has no fetch, so we provide a fake
});

describe("normalizeResponse", () => {
  test("maps Person-2 field names (claim_text / evidence) to UI shape", () => {
    const r = normalizeResponse(okBody);
    expect(r.claims[0].text).toBe("The Earth orbits the Sun.");
    expect(r.claims[0].citations[0]).toMatchObject({ sourceId: "KB-001" });
    expect(r.processingTimeMs).toBe(120);
  });
  test("handles the SRS field names (text / citations)", () => {
    const r = normalizeResponse({
      claims: [{ text: "X is Y.", verdict: "Unverifiable", confidence: 0.5, citations: [] }],
    });
    expect(r.claims[0].text).toBe("X is Y.");
  });
  test("empty claims becomes no_verifiable_claims", () => {
    expect(normalizeResponse({ claims: [] }).status).toBe("no_verifiable_claims");
  });
});

describe("verifyText", () => {
  test("POSTs to /verify with JSON body and API key header", async () => {
    global.fetch.mockResolvedValue({ ok: true, json: async () => okBody });
    const fetchMock = global.fetch;
    await verifyText("hello world is here", { apiKey: "abc" });
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/verify$/);
    expect(opts.method).toBe("POST");
    expect(opts.headers["X-API-Key"]).toBe("abc");
    expect(JSON.parse(opts.body)).toEqual({ text: "hello world is here" });
  });
  test("401 gives a helpful API-key message", async () => {
    global.fetch.mockResolvedValue({ ok: false, status: 401, json: async () => ({}) });
    await expect(verifyText("x is y")).rejects.toThrow(/API key/i);
  });
  test("400 includes server detail", async () => {
    global.fetch.mockResolvedValue({ ok: false, status: 400, json: async () => ({ detail: "Too long" }) });
    await expect(verifyText("x is y")).rejects.toThrow(/Too long/);
  });
  test("network failure gives a friendly ApiError", async () => {
    global.fetch.mockRejectedValue(new TypeError("fetch failed"));
    const err = await verifyText("x is y").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/Cannot reach/);
  });
});
