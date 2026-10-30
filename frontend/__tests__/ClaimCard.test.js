import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ClaimCard from "../src/components/ClaimCard";

const base = { id: 1, text: "The Earth orbits the Sun.", verdict: "Supported", confidence: 0.97, citations: [] };

test("shows claim text, verdict and confidence", () => {
  render(<ClaimCard claim={base} />);
  expect(screen.getByText("The Earth orbits the Sun.")).toBeInTheDocument();
  expect(screen.getByText("Supported")).toBeInTheDocument();
  expect(screen.getByText("97% confidence")).toBeInTheDocument();
});

test.each([
  ["Supported", "supported"],
  ["Contradicted", "contradicted"],
  ["Unverifiable", "unverifiable"],
])("%s uses the %s colour class (REQ-16)", (verdict, cls) => {
  render(<ClaimCard claim={{ ...base, verdict }} />);
  expect(screen.getByTestId("claim-card")).toHaveClass(cls);
});

test("citations are hidden until expanded", async () => {
  const claim = { ...base, citations: [{ sourceId: "KB-001", text: "Earth orbits the Sun yearly.", relevance: 0.8 }] };
  render(<ClaimCard claim={claim} />);
  expect(screen.queryByText("Earth orbits the Sun yearly.")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /show evidence/i }));
  expect(screen.getByText("Earth orbits the Sun yearly.")).toBeInTheDocument();
  expect(screen.getByText(/KB-001/)).toBeInTheDocument();
});

test("claim without citations shows a no-evidence note", () => {
  render(<ClaimCard claim={{ ...base, verdict: "Unverifiable" }} />);
  expect(screen.getByText(/no supporting evidence/i)).toBeInTheDocument();
});

// ---- Sprint 2 (#19 / #20)
test("verdict is shown as an icon + word, not by colour alone", () => {
  const { rerender } = render(<ClaimCard claim={base} />);
  expect(screen.getByText("✓")).toBeInTheDocument();
  rerender(<ClaimCard claim={{ ...base, verdict: "Contradicted" }} />);
  expect(screen.getByText("✗")).toBeInTheDocument();
  rerender(<ClaimCard claim={{ ...base, verdict: "Unverifiable" }} />);
  expect(screen.getByText("?")).toBeInTheDocument();
});

test("shows the source sentence only when it differs from the claim", () => {
  const { rerender } = render(<ClaimCard claim={{ ...base, sourceSentence: "The Earth orbits the Sun." }} />);
  expect(screen.queryByText(/^From:/)).not.toBeInTheDocument();
  rerender(<ClaimCard claim={{ ...base, sourceSentence: "Scientists say the Earth orbits the Sun, and so on." }} />);
  expect(screen.getByText(/From: .*Scientists say/)).toBeInTheDocument();
});

test("citation shows its relevance score", async () => {
  const claim = { ...base, citations: [{ sourceId: "KB-001", text: "Earth orbits the Sun yearly.", relevance: 0.81 }] };
  render(<ClaimCard claim={claim} />);
  await userEvent.click(screen.getByRole("button", { name: /show evidence/i }));
  expect(screen.getByText(/relevance 81%/)).toBeInTheDocument();
});
