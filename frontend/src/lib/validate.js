export const MAX_WORDS = 2000; // SRS REQ-1 / REQ-15

const WORD_RE = /[\p{L}\p{N}_]+(?:['’-][\p{L}\p{N}_]+)*/gu;

export function countWords(text) {
  return (text.match(WORD_RE) || []).length;
}

// Returns an error message string, or null if the input is valid.
export function validateInput(text) {
  if (!text || !text.trim()) return "Please enter some text to verify.";
  const words = countWords(text);
  if (words > MAX_WORDS) {
    return `Your text has ${words} words. The maximum allowed is ${MAX_WORDS}.`;
  }
  return null;
}
