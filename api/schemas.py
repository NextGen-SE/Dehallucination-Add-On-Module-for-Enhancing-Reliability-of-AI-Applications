"""Request/response schemas for POST /verify (SRS Appendix B).

Appendix B is the canonical contract. Where it is silent, fields are taken
from the existing modules rather than invented:
- `claim_id`, `source_sentence` and the offsets come from `claim_extraction.Claim`.
- `status` / `message` come from `claim_extraction.ExtractionResult` and carry
  the REQ-4 "No verifiable claims found" result.
- citations use `evidence_verification.EvidencePassage` field names.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictStr

# Must match evidence_verification.VERDICT_* (checked in tests/test_api.py).
Verdict = Literal["Supported", "Contradicted", "Unverifiable"]


class VerifyRequest(BaseModel):
    # StrictStr: reject numbers/objects instead of coercing them (REQ-15).
    # The 2,000-word limit (REQ-1) is enforced by claim_extraction itself.
    text: StrictStr = Field(description="AI-generated text to verify (<= 2000 words).")
    knowledge_base_id: Optional[StrictStr] = Field(
        default=None,
        max_length=64,
        description='Knowledge base to verify against; defaults to the configured KB ("wikipedia_subset").',
    )


class Span(BaseModel):
    """Half-open character offsets [start_char, end_char) into the request text."""

    start_char: int
    end_char: int


class Citation(BaseModel):
    source_id: str
    text: str
    relevance_score: float


class NLIScores(BaseModel):
    entailment: float
    contradiction: float
    neutral: float


class EvidenceAssessment(BaseModel):
    """A passage the NLI model assessed for this claim (REQ-1.2: up to 3)."""

    source_id: str
    text: str
    relevance_score: float
    nli_label: Literal["entailment", "contradiction", "neutral"]
    nli_confidence: float = Field(ge=0.0, le=1.0, description="Probability of nli_label.")
    nli_scores: NLIScores
    supports_verdict: bool = Field(description="True if this passage is one of the claim's citations.")


class ClaimResult(BaseModel):
    claim_id: int
    text: str
    span: Span
    source_sentence: str
    source_sentence_span: Span
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    # Passages that justify the verdict (empty for Unverifiable), strongest first.
    citations: list[Citation]
    # Every passage the NLI model assessed, most relevant first; may be fewer
    # than 3 (only passages above the relevance threshold are assessed) and is
    # empty when none qualifies.
    evidence_passages: list[EvidenceAssessment] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    status: Literal["ok", "no_verifiable_claims"]
    message: Optional[str] = None
    claims: list[ClaimResult]
    processing_time_ms: int


class ErrorResponse(BaseModel):
    detail: str
