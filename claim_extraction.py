
"""TruthLayer candidate claim extraction: a standard-library baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

MAX_WORDS = 2000

_WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", re.UNICODE)

_SENTENCE_RE = re.compile(
    r".+?(?:[.!?]+(?=\s|$)|$)",
    re.DOTALL,
)

_OPINION_START_RE = re.compile(
    r"^\s*(?:in my opinion\b|i think\b|i believe\b|i feel\b|"
    r"i prefer\b|i love\b|i hate\b|in my view\b|"
    r"what do you think\b|why not\b|how about\b)",
    re.IGNORECASE,
)

_SUBJECTIVE_RE = re.compile(
    r"\b(?:beautiful|ugly|wonderful|terrible|amazing|awful|"
    r"best|worst|boring|delicious|fantastic|horrible)\b",
    re.IGNORECASE,
)

_COMMAND_START_RE = re.compile(
    r"^\s*(?:please\s+)?(?:do|don't|dont|let's|lets|consider|"
    r"imagine|tell|show|explain|look|remember|note)\b",
    re.IGNORECASE,
)

_STATEMENT_VERBS_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|will|would|"
    r"does|do|did|contains|causes|increases|decreases|orbits|"
    r"located|founded|produces|uses|means|equals|runs|lives|"
    r"works|grows|flows|moves|occurs|forms|consists|includes|"
    r"reduces|improves)\b",
    re.IGNORECASE,
)

# Conservative pattern for conjunctions that may join independent clauses.
_ATOMIC_SEPARATOR_RE = re.compile(
    r"\s*,?\s+\band\b\s+",
    re.IGNORECASE,
)


class InputTooLongError(ValueError):
    """Raised when input exceeds the configured maximum word count."""


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    source_sentence: str
    start_char: int
    end_char: int
    source_sentence_start: int
    source_sentence_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionResult:
    status: str
    claims: list[Claim]
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "status": self.status,
            "claims": [claim.to_dict() for claim in self.claims],
        }
        if self.message:
            output["message"] = self.message
        return output


def count_words(text: str) -> int:
    """Count words using a Unicode-aware regular expression."""
    return len(_WORD_RE.findall(text))


def _sentences_with_offsets(
    text: str,
) -> list[tuple[str, int, int]]:
    """Split text into sentence-like chunks while retaining source offsets."""
    chunks: list[tuple[str, int, int]] = []

    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)

        if not raw.strip():
            continue

        leading = len(raw) - len(raw.lstrip())
        trimmed = raw.rstrip()

        if not trimmed:
            continue

        start = match.start() + leading
        end = match.start() + len(trimmed)

        chunks.append((text[start:end], start, end))

    return chunks


def _looks_like_candidate_fact(sentence: str) -> bool:
    """Apply conservative surface heuristics, not semantic fact detection."""
    sentence = sentence.strip()

    if not sentence or sentence.endswith("?"):
        return False

    if (
        _OPINION_START_RE.search(sentence)
        or _COMMAND_START_RE.search(sentence)
    ):
        return False

    if _SUBJECTIVE_RE.search(sentence):
        return False

    if len(_WORD_RE.findall(sentence)) < 3:
        return False

    return bool(_STATEMENT_VERBS_RE.search(sentence))


def _atomic_claim_spans(
    sentence: str,
    sentence_start: int,
) -> list[tuple[str, int, int]]:
    """Split common conjunction-linked factual clauses and retain offsets.

    Splitting occurs only when every resulting clause independently passes
    the candidate-fact heuristic. Otherwise, the original sentence is kept.
    """
    matches = list(_ATOMIC_SEPARATOR_RE.finditer(sentence))

    if not matches:
        return [
            (sentence, sentence_start, sentence_start + len(sentence))
        ]

    spans: list[tuple[str, int, int]] = []
    previous_end = 0

    for match in matches:
        left = sentence[previous_end:match.start()].strip()

        if not left:
            return [
                (sentence, sentence_start, sentence_start + len(sentence))
            ]

        left_start = sentence.find(left, previous_end)
        left_end = left_start + len(left)

        spans.append((
            left,
            sentence_start + left_start,
            sentence_start + left_end,
        ))

        previous_end = match.end()

    right = sentence[previous_end:].strip()

    if not right:
        return [
            (sentence, sentence_start, sentence_start + len(sentence))
        ]

    right_start = sentence.find(right, previous_end)
    right_end = right_start + len(right)

    spans.append((
        right,
        sentence_start + right_start,
        sentence_start + right_end,
    ))

    if all(
        _looks_like_candidate_fact(claim_text)
        for claim_text, _, _ in spans
    ):
        return spans

    return [
        (sentence, sentence_start, sentence_start + len(sentence))
    ]


def extract_claims(
    text: str,
    max_words: int = MAX_WORDS,
) -> ExtractionResult:
    """Extract candidate claims and preserve their original source mapping.

    Raises:
        TypeError: Input is not a string or max_words is invalid.
        InputTooLongError: Input exceeds max_words.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if (
        not isinstance(max_words, int)
        or isinstance(max_words, bool)
        or max_words < 1
    ):
        raise TypeError("max_words must be a positive integer")

    if not text.strip():
        return ExtractionResult(
            "no_verifiable_claims",
            [],
            "No verifiable claims found",
        )

    word_count = count_words(text)

    if word_count > max_words:
        raise InputTooLongError(
            f"Input contains {word_count} words; "
            f"maximum allowed is {max_words}."
        )

    claims: list[Claim] = []

    for sentence, start, end in _sentences_with_offsets(text):
        if not _looks_like_candidate_fact(sentence):
            continue

        atomic_spans = _atomic_claim_spans(sentence, start)

        for claim_text, claim_start, claim_end in atomic_spans:
            claims.append(
                Claim(
                    claim_id=len(claims) + 1,
                    text=claim_text,
                    source_sentence=sentence,
                    start_char=claim_start,
                    end_char=claim_end,
                    source_sentence_start=start,
                    source_sentence_end=end,
                )
            )

    if not claims:
        return ExtractionResult(
            "no_verifiable_claims",
            [],
            "No verifiable claims found",
        )

    return ExtractionResult("ok", claims)


if __name__ == "__main__":
    sample = (
        "The Earth orbits the Sun. "
        "Water contains hydrogen and oxygen. "
        "Bengaluru is in Karnataka, and Karnataka is in India. "
        "I think this is wonderful. "
        "Is the Moon made of cheese?"
    )

    print(extract_claims(sample).to_dict())
