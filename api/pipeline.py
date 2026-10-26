"""Glue between Person 1's extractor and Person 2's verifier.

Uses only the modules' public interfaces:
    extract_claims(text) -> ExtractionResult(status, claims, message)
    EvidenceVerifier.verify_claims(claims) -> list[VerificationResult]
and merges the two results by `claim_id` into the Appendix B shape.
"""
from __future__ import annotations

import threading
from collections import Counter

from claim_extraction import extract_claims
from evidence_verification import EvidenceVerifier

from api.schemas import Citation, ClaimResult, Span


class VerificationPipeline:
    def __init__(self, verifier: EvidenceVerifier) -> None:
        self._verifier = verifier
        # NFR-2 allows queuing; serialising model calls avoids relying on the
        # thread-safety of the underlying HuggingFace models.
        self._lock = threading.Lock()

    def run(self, text: str) -> tuple[str, str | None, list[ClaimResult]]:
        """Return (status, message, claims).

        Raises claim_extraction.InputTooLongError if text exceeds REQ-1's limit.
        """
        extraction = extract_claims(text)
        if not extraction.claims:
            return extraction.status, extraction.message, []

        with self._lock:
            verifications = self._verifier.verify_claims(extraction.claims)
        by_id = {v.claim_id: v for v in verifications}

        claims = []
        for claim in extraction.claims:
            v = by_id[claim.claim_id]
            claims.append(
                ClaimResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    span=Span(start_char=claim.start_char, end_char=claim.end_char),
                    source_sentence=claim.source_sentence,
                    source_sentence_span=Span(
                        start_char=claim.source_sentence_start,
                        end_char=claim.source_sentence_end,
                    ),
                    verdict=v.verdict,
                    confidence=v.confidence,
                    citations=[
                        Citation(
                            source_id=e.source_id,
                            text=e.text,
                            relevance_score=e.relevance_score,
                        )
                        for e in v.evidence
                    ],
                )
            )
        return extraction.status, extraction.message, claims


def verdict_summary(claims: list[ClaimResult]) -> dict[str, int]:
    counts = Counter(c.verdict for c in claims)
    return {
        "supported": counts.get("Supported", 0),
        "contradicted": counts.get("Contradicted", 0),
        "unverifiable": counts.get("Unverifiable", 0),
    }
