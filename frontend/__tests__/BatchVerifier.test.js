import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import BatchVerifier from "../src/components/BatchVerifier";
import Home from "../src/app/page";
import { verifyText, ApiError } from "../src/lib/api";
import { downloadFile } from "../src/lib/report";

jest.mock("../src/lib/api", () => ({ ...jest.requireActual("../src/lib/api"), verifyText: jest.fn() }));
jest.mock("../src/lib/report", () => ({ ...jest.requireActual("../src/lib/report"), downloadFile: jest.fn() }));

const ok = (text) => ({
  status: "ok", message: null, processingTimeMs: 10,
  claims: [{ id: 1, text, verdict: "Supported", confidence: 0.9, citations: [] }],
});

beforeEach(() => { verifyText.mockReset(); downloadFile.mockReset(); });

// userEvent.type treats { and [ as special; paste is safer for multi-line input.
async function enter(value) {
  const box = screen.getByLabelText(/texts to verify/i);
  await userEvent.click(box);
  await userEvent.paste(value);
}

test("empty batch shows a validation error and calls nothing", async () => {
  render(<BatchVerifier apiKey="k" />);
  await userEvent.click(screen.getByRole("button", { name: "Verify batch" }));
  expect(screen.getByRole("alert")).toHaveTextContent(/at least one/i);
  expect(verifyText).not.toHaveBeenCalled();
});

test("more than 50 items is rejected with a clear message", async () => {
  render(<BatchVerifier apiKey="k" />);
  await enter(Array.from({ length: 51 }, (_, i) => `Fact ${i} is true.`).join("\n---\n"));
  expect(screen.getByText("51 / 50 items")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Verify batch" }));
  expect(screen.getByRole("alert")).toHaveTextContent(/maximum is 50/);
  expect(verifyText).not.toHaveBeenCalled();
});

test("shows each item's status separately, including a partial failure", async () => {
  verifyText.mockImplementation(async (t) => {
    if (t.startsWith("Second")) throw new ApiError("The verification pipeline failed on the server.", 500);
    return ok(t);
  });
  render(<BatchVerifier apiKey="k" />);
  await enter("First is true.\n---\nSecond is true.\n---\nThird is true.");
  await userEvent.click(screen.getByRole("button", { name: "Verify batch" }));

  const rows = await screen.findAllByTestId("batch-item");
  expect(rows).toHaveLength(3);
  await screen.findByText(/3 of 3 finished/);
  expect(within(rows[0]).getByText("Done")).toBeInTheDocument();
  expect(within(rows[1]).getByText("Failed")).toBeInTheDocument();
  expect(within(rows[1]).getByText(/pipeline failed/)).toBeInTheDocument();
  expect(within(rows[2]).getByText("Done")).toBeInTheDocument();
  expect(verifyText).toHaveBeenCalledWith("First is true.", { apiKey: "k" });
});

test("report buttons appear after the batch and download HTML, CSV and JSON", async () => {
  verifyText.mockImplementation(async (t) => ok(t));
  render(<BatchVerifier apiKey="" />);
  await enter("One is true.\n---\nTwo is true.");
  await userEvent.click(screen.getByRole("button", { name: "Verify batch" }));
  await screen.findByRole("group", { name: /download report/i });

  await userEvent.click(screen.getByRole("button", { name: "HTML" }));
  await userEvent.click(screen.getByRole("button", { name: "CSV" }));
  await userEvent.click(screen.getByRole("button", { name: "JSON" }));
  expect(downloadFile).toHaveBeenCalledTimes(3);
  const [name, content] = downloadFile.mock.calls[0];
  expect(name).toMatch(/^truthlayer-report-\d{8}-\d{6}\.html$/);
  expect(content).toContain("Item 1");
  expect(content).toContain("Item 2");
});

test("page: single-text results offer a report download too, and the tabs switch", async () => {
  verifyText.mockResolvedValue(ok("The Earth orbits the Sun."));
  render(<Home />);
  await userEvent.type(screen.getByLabelText(/^text to verify/i), "The Earth orbits the Sun.");
  await userEvent.click(screen.getByRole("button", { name: "Verify" }));
  await screen.findAllByTestId("claim-card");
  await userEvent.click(screen.getByRole("button", { name: "HTML" }));
  expect(downloadFile).toHaveBeenCalledTimes(1);

  expect(screen.getByRole("tab", { name: "Single text" })).toHaveAttribute("aria-selected", "true");
  await userEvent.click(screen.getByRole("tab", { name: /batch/i }));
  expect(screen.getByRole("tab", { name: /batch/i })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("button", { name: "Verify batch" })).toBeVisible();
});
