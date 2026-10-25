import { countWords, validateInput, MAX_WORDS } from "../src/lib/validate";

describe("validate", () => {
  test("counts words, treating hyphenated words as one", () => {
    expect(countWords("The Earth orbits the Sun.")).toBe(5);
    expect(countWords("state-of-the-art model")).toBe(2);
  });
  test("empty / whitespace input is invalid", () => {
    expect(validateInput("")).toMatch(/enter some text/i);
    expect(validateInput("   \n ")).toMatch(/enter some text/i);
  });
  test("input over the limit is invalid", () => {
    const big = "word ".repeat(MAX_WORDS + 1);
    expect(validateInput(big)).toMatch(/maximum allowed is 2000/);
  });
  test("normal input is valid", () => {
    expect(validateInput("Water contains hydrogen and oxygen.")).toBeNull();
  });
});
