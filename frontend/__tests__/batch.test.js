import { parseBatchInput, validateBatch, runBatch, MAX_BATCH_ITEMS } from "../src/lib/batch";
import { ApiError } from "../src/lib/api";

const items = (n) => Array.from({ length: n }, (_, i) => ({ id: i + 1, text: `Fact number ${i + 1} is true.` }));

describe("parseBatchInput", () => {
  test("splits on a line containing only ---, trims and drops empties", () => {
    const parsed = parseBatchInput("First text.\n---\n  Second text.  \n\n---\n\n---\nThird text.");
    expect(parsed.map((p) => p.text)).toEqual(["First text.", "Second text.", "Third text."]);
    expect(parsed.map((p) => p.id)).toEqual([1, 2, 3]);
  });
  test("a blank line alone does not split an item; CRLF works", () => {
    expect(parseBatchInput("Para one.\n\nPara two.")).toHaveLength(1);
    expect(parseBatchInput("A.\r\n---\r\nB.")).toHaveLength(2);
  });
  test("empty input gives no items", () => {
    expect(parseBatchInput("  \n ")).toEqual([]);
  });
});

describe("validateBatch", () => {
  test("rejects empty batch and more than 50 items, accepts exactly 50", () => {
    expect(validateBatch([])).toMatch(/at least one/i);
    expect(validateBatch(items(MAX_BATCH_ITEMS + 1))).toMatch(/maximum is 50/);
    expect(validateBatch(items(MAX_BATCH_ITEMS))).toBeNull();
  });
});

describe("runBatch", () => {
  test("verifies every item and reports each result separately", async () => {
    const verify = jest.fn(async (t) => ({ status: "ok", claims: [], echo: t }));
    const rows = await runBatch(items(4), verify);
    expect(rows.map((r) => r.status)).toEqual(["done", "done", "done", "done"]);
    expect(verify).toHaveBeenCalledTimes(4);
  });

  test("partial failure: one failing item does not stop the others", async () => {
    const verify = jest.fn(async (t) => {
      if (t.includes("number 2")) throw new ApiError("Server error", 500);
      return { status: "ok", claims: [] };
    });
    const rows = await runBatch(items(3), verify);
    expect(rows.map((r) => r.status)).toEqual(["done", "failed", "done"]);
    expect(rows[1].error).toBe("Server error");
  });

  test("invalid items (empty or over 2000 words) are not sent to the API", async () => {
    const verify = jest.fn(async () => ({ status: "ok", claims: [] }));
    const big = Array(2001).fill("word").join(" ");
    const rows = await runBatch([{ id: 1, text: "Good text." }, { id: 2, text: "   " }, { id: 3, text: big }], verify);
    expect(rows.map((r) => r.status)).toEqual(["done", "invalid", "invalid"]);
    expect(verify).toHaveBeenCalledTimes(1);
  });

  test("a 401 stops new requests; remaining items are skipped", async () => {
    const verify = jest.fn(async () => {
      throw new ApiError("bad key", 401);
    });
    const rows = await runBatch(items(6), verify, { concurrency: 1 });
    expect(verify).toHaveBeenCalledTimes(1);
    expect(rows[0].status).toBe("failed");
    expect(rows.slice(1).every((r) => r.status === "skipped")).toBe(true);
  });

  test("never runs more requests at once than the concurrency limit", async () => {
    let active = 0;
    let peak = 0;
    const verify = async () => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 5));
      active--;
      return { status: "ok", claims: [] };
    };
    await runBatch(items(10), verify, { concurrency: 3 });
    expect(peak).toBeLessThanOrEqual(3);
    expect(peak).toBeGreaterThan(1);
  });

  test("calls onUpdate as items change state", async () => {
    const seen = [];
    await runBatch(items(1), async () => ({ status: "ok", claims: [] }), { onUpdate: (i, r) => seen.push(r.status) });
    expect(seen).toEqual(["running", "done"]);
  });
});
