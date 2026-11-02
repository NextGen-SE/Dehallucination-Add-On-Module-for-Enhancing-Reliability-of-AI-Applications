// Batch verification (Issue #22).
// The backend only has POST /verify, so a batch is run in the browser:
// each item is sent as its own /verify request, a few at a time.
import { countWords, validateInput } from "./validate";

export const MAX_BATCH_ITEMS = 50;
export const BATCH_CONCURRENCY = 3;

// Items are separated by a line that contains only "---".
const SEPARATOR = /^[ \t]*-{3,}[ \t]*$/m;

export function parseBatchInput(raw) {
  return (raw || "")
    .split(SEPARATOR)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((text, i) => ({ id: i + 1, text, words: countWords(text) }));
}

// Returns an error message, or null if the batch can be run.
export function validateBatch(items) {
  if (items.length === 0) return "Please enter at least one text to verify.";
  if (items.length > MAX_BATCH_ITEMS) {
    return `You entered ${items.length} items. The maximum is ${MAX_BATCH_ITEMS} per batch.`;
  }
  return null;
}

// Statuses: pending -> running -> done | failed, or invalid / skipped.
// One failing item never stops the others, except a 401 (bad API key),
// which would fail every remaining item, so those are skipped instead.
export async function runBatch(items, verifyFn, { concurrency = BATCH_CONCURRENCY, onUpdate = () => {} } = {}) {
  const rows = items.map((it) => ({ id: it.id, status: "pending" }));
  const update = (i, patch) => {
    rows[i] = { ...rows[i], ...patch };
    onUpdate(i, rows[i]);
  };

  items.forEach((it, i) => {
    const problem = validateInput(it.text);
    if (problem) update(i, { status: "invalid", error: problem });
  });

  let next = 0;
  let stopReason = null;

  async function worker() {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      if (rows[i].status === "invalid") continue;
      if (stopReason) {
        update(i, { status: "skipped", error: stopReason });
        continue;
      }
      update(i, { status: "running" });
      try {
        update(i, { status: "done", result: await verifyFn(items[i].text) });
      } catch (e) {
        update(i, { status: "failed", error: (e && e.message) || "Verification failed." });
        if (e && e.status === 401) stopReason = "Skipped because the API key was rejected.";
      }
    }
  }

  const workers = Math.max(1, Math.min(concurrency, items.length));
  await Promise.all(Array.from({ length: workers }, worker));
  return rows;
}
