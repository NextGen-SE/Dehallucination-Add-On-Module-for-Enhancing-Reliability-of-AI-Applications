"""TruthLayer evidence retrieval and NLI-based claim verification.

Consumes the Claim objects produced by Person 1's `claim_extraction.py`,
retrieves the most relevant passage(s) from a knowledge base, and uses an
NLI (Natural Language Inference) model to decide whether each claim is
Supported, Contradicted, or Unverifiable — per SRS REQ-5 through REQ-11.

Design notes
------------
The embedding model and the NLI classifier are both injected as callables
(`Embedder` / `NLIClassifier` protocols below) rather than hard-wired to a
specific library call inside the verification logic. This keeps the
decision logic (retrieval ranking, thresholding, verdict mapping) fully
unit-testable with lightweight fakes, with no network access or model
download required to verify correctness. The real implementations
(`SentenceTransformerEmbedder`, `TransformersNLIClassifier`) are thin
adapters around `sentence-transformers` / `transformers` and are only
exercised by the integration test, which is skipped automatically if the
models aren't available (e.g. no internet access).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_TOP_K = 5
DEFAULT_RELEVANCE_THRESHOLD = 0.35

VERDICT_SUPPORTED = "Supported"
VERDICT_CONTRADICTED = "Contradicted"
VERDICT_UNVERIFIABLE = "Unverifiable"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvidencePassage:
    source_id: str
    text: str
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    claim_text: str
    verdict: str
    confidence: float
    evidence: list[EvidencePassage]
    claim_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }


# --------------------------------------------------------------------------- #
# Pluggable model interfaces (what makes this testable without downloads)
# --------------------------------------------------------------------------- #

class Embedder(Protocol):
    """Callable: list[str] -> list[list[float]] (one embedding vector per text)."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class NLIClassifier(Protocol):
    """Callable: (premise, hypothesis) -> {'entailment': f, 'contradiction': f, 'neutral': f}.

    The three scores must be non-negative and should sum to ~1.0.
    """

    def __call__(self, premise: str, hypothesis: str) -> dict[str, float]: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------- #
# Knowledge base loading
# --------------------------------------------------------------------------- #

def load_knowledge_base(path: str | Path) -> list[dict[str, str]]:
    """Load a knowledge base from a JSON file.

    Expected format: a list of {"source_id": str, "text": str} objects.
    Swapping the knowledge base (REQ-9) just means pointing this at a
    different JSON file with the same shape.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(
        isinstance(item, dict) and "source_id" in item and "text" in item
        for item in data
    ):
        raise ValueError(
            "Knowledge base file must be a JSON list of "
            '{"source_id": ..., "text": ...} objects.'
        )
    return data


# --------------------------------------------------------------------------- #
# Core verifier
# --------------------------------------------------------------------------- #

class EvidenceVerifier:
    """Retrieves evidence and classifies claims as Supported/Contradicted/Unverifiable.

    Parameters
    ----------
    knowledge_base:
        List of {"source_id": str, "text": str} passages (see load_knowledge_base).
    embedder:
        Callable producing one embedding vector per input text. If omitted,
        a real `SentenceTransformerEmbedder` is constructed lazily (requires
        `sentence-transformers` and internet access on first use).
    nli_classifier:
        Callable scoring (premise, hypothesis) pairs. If omitted, a real
        `TransformersNLIClassifier` is constructed lazily (requires
        `transformers` and internet access on first use).
    top_k:
        Number of passages to retrieve per claim (REQ-5).
    relevance_threshold:
        Minimum cosine similarity a passage must reach to be considered
        usable evidence at all (REQ-8). Tune this empirically once you
        have real embeddings; 0.35 is a starting point, not a measured value.
    """

    def __init__(
        self,
        knowledge_base: list[dict[str, str]],
        embedder: Embedder | None = None,
        nli_classifier: NLIClassifier | None = None,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        if not knowledge_base:
            raise ValueError("knowledge_base must not be empty")

        self._kb = knowledge_base
        self._embed = embedder or SentenceTransformerEmbedder()
        self._classify = nli_classifier or TransformersNLIClassifier()
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

        kb_texts = [item["text"] for item in self._kb]
        self._kb_embeddings = self._embed(kb_texts)

    def retrieve(
        self, claim_text: str, top_k: int | None = None
    ) -> list[EvidencePassage]:
        """Return the top_k most relevant knowledge-base passages, best first."""
        k = top_k if top_k is not None else self.top_k
        (claim_embedding,) = self._embed([claim_text])

        scored = [
            EvidencePassage(
                source_id=item["source_id"],
                text=item["text"],
                relevance_score=round(
                    _cosine_similarity(claim_embedding, kb_emb), 4
                ),
            )
            for item, kb_emb in zip(self._kb, self._kb_embeddings)
        ]
        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored[:k]

    def verify_claim(
        self, claim_text: str, claim_id: int | None = None
    ) -> VerificationResult:
        """Retrieve evidence and classify a single claim (REQ-6, REQ-7, REQ-8, REQ-10)."""
        passages = self.retrieve(claim_text)

        if not passages or passages[0].relevance_score < self.relevance_threshold:
            best = passages[0].relevance_score if passages else 0.0
            return VerificationResult(
                claim_text=claim_text,
                verdict=VERDICT_UNVERIFIABLE,
                confidence=round(max(0.0, 1.0 - best), 3),
                evidence=[],
                claim_id=claim_id,
            )

        best_passage = passages[0]
        scores = self._classify(premise=best_passage.text, hypothesis=claim_text)
        predicted_label = max(scores, key=scores.get)
        confidence = round(scores[predicted_label], 3)

        if predicted_label == "entailment":
            verdict = VERDICT_SUPPORTED
            evidence = [best_passage]
        elif predicted_label == "contradiction":
            verdict = VERDICT_CONTRADICTED
            evidence = [best_passage]
        else:  # "neutral" — relevant passage found, but it neither confirms nor denies
            verdict = VERDICT_UNVERIFIABLE
            evidence = []

        return VerificationResult(
            claim_text=claim_text,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            claim_id=claim_id,
        )

    def verify_claims(self, claims: Iterable[Any]) -> list[VerificationResult]:
        """Verify many claims. Accepts Person 1's `Claim` objects or plain strings.

        A `Claim` object is recognised by duck typing: anything with a
        `.text` attribute is treated as one, and its `.claim_id` is carried
        through if present. Plain strings are also accepted directly, which
        is convenient for standalone testing of this module.
        """
        results = []
        for claim in claims:
            if isinstance(claim, str):
                results.append(self.verify_claim(claim))
            else:
                text = getattr(claim, "text")
                claim_id = getattr(claim, "claim_id", None)
                results.append(self.verify_claim(text, claim_id=claim_id))
        return results


# --------------------------------------------------------------------------- #
# Real model adapters (network/model-download required on first use)
# --------------------------------------------------------------------------- #

class SentenceTransformerEmbedder:
    """Wraps sentence-transformers for semantic embedding (REQ-5)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self._model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class TransformersNLIClassifier:
    """Wraps a HuggingFace NLI model via the text-classification pipeline (REQ-6).

    Uses the pipeline (not a hardcoded label-index order) so the premise/
    contradiction/entailment/neutral label names come straight from the
    model's own config (`id2label`), rather than from an assumption about
    label ordering that could be wrong for a given checkpoint.
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        from transformers import pipeline  # local import: optional dep

        self._pipe = pipeline("text-classification", model=model_name, top_k=None)

    def __call__(self, premise: str, hypothesis: str) -> dict[str, float]:
        raw = self._pipe({"text": premise, "text_pair": hypothesis})
        # raw is a list of {"label": ..., "score": ...} for every class
        scores = {item["label"].lower(): float(item["score"]) for item in raw}
        for expected in ("entailment", "contradiction", "neutral"):
            if expected not in scores:
                raise RuntimeError(
                    f"NLI model output missing expected label '{expected}'. "
                    f"Got labels: {list(scores)}. Check the model's id2label config."
                )
        return scores
