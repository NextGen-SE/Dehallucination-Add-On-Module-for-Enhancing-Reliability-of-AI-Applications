import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Home from "../src/app/page";
import { verifyText, ApiError } from "../src/lib/api";

jest.mock("../src/lib/api", () => ({
  ...jest.requireActual("../src/lib/api"),
  verifyText: jest.fn(),
}));

const result = {
  status: "ok",
  message: null,
  processingTimeMs: 50,
  claims: [
    { id: 1, text: "The Earth orbits the Sun.", verdict: "Supported", confidence: 0.9, citations: [] },
    { id: 2, text: "The Sun orbits the Earth.", verdict: "Contradicted", confidence: 0.8, citations: [] },
  ],
};

beforeEach(() => verifyText.mockReset());

test("empty input shows a validation error and does not call the API", async () => {
  render(<Home />);
  await userEvent.click(screen.getByRole("button", { name: "Verify" }));
  expect(screen.getByRole("alert")).toHaveTextContent(/enter some text/i);
  expect(verifyText).not.toHaveBeenCalled();
});

test("shows a loading indicator while waiting, then the results", async () => {
  let resolve;
  verifyText.mockReturnValue(new Promise((r) => (resolve = r)));
  render(<Home />);
  await userEvent.type(screen.getByLabelText(/text to verify/i), "The Earth orbits the Sun.");
  await userEvent.click(screen.getByRole("button", { name: "Verify" }));
  expect(screen.getByRole("status")).toHaveTextContent(/verifying/i);

  resolve(result);
  expect(await screen.findAllByTestId("claim-card")).toHaveLength(2);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  expect(screen.getByText(/1 supported, 1\s+contradicted/)).toBeInTheDocument();
});

test("API failure shows the error banner", async () => {
  verifyText.mockRejectedValue(new ApiError("Cannot reach the TruthLayer server.", 0));
  render(<Home />);
  await userEvent.type(screen.getByLabelText(/text to verify/i), "Water is wet.");
  await userEvent.click(screen.getByRole("button", { name: "Verify" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach/);
});

test("'no verifiable claims' result shows a friendly message, not an error", async () => {
  verifyText.mockResolvedValue({ status: "no_verifiable_claims", message: "No verifiable claims found", claims: [] });
  render(<Home />);
  await userEvent.type(screen.getByLabelText(/text to verify/i), "I think this is wonderful.");
  await userEvent.click(screen.getByRole("button", { name: "Verify" }));
  expect(await screen.findByText("No verifiable claims found")).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

test("disclaimer is always visible (NFR-5)", () => {
  render(<Home />);
  expect(screen.getByText(/does not eliminate/i)).toBeInTheDocument();
});
