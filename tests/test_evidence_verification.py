import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_verification import (
    EvidenceVerifier,
    VERDICT_CONTRADICTED,
    VERDICT_SUPPORTED,
    VERDICT_UNVERIFIABLE,
    load_knowledge_base,
)

KB_PATH = Path(__file__).resolve().parents[1] / "knowledge_base.json"


# --------------------------------------------------------------------------- #
# Deterministic fakes — no model download, no network, fully reproducible.
# --------------------------------------------------------------------------- #

class FakeEmbedder:
    """Returns a hand-assigned vector per known text; unknown text -> zero vector."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0, 0.0]) for t in texts]


class FakeNLIClassifier:
    """Returns a hand-assigned score dict per (premise, hypothesis) pair."""

    def __init__(self, answers: dict[tuple[str, str], dict[str, float]]) -> None:
        self._answers = answers

    def __call__(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self._answers[(premise, hypothesis)]


class TestLoadKnowledgeBase(unittest.TestCase):
    def test_loads_real_knowledge_base_file(self):
        kb = load_knowledge_base(KB_PATH)
        self.assertGreaterEqual(len(kb), 10)
        self.assertIn("source_id", kb[0])
        self.assertIn("text", kb[0])

    def test_rejects_malformed_file(self, tmp_path=Path("/tmp")):
        bad_file = tmp_path / "bad_kb.json"
        bad_file.write_text('["not", "the right", "shape"]', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_knowledge_base(bad_file)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.kb = [
            {"source_id": "KB-A", "text": "passage about cats"},
            {"source_id": "KB-B", "text": "passage about dogs"},
            {"source_id": "KB-C", "text": "passage about the solar system"},
        ]
        # Orthogonal-ish vectors so similarity ranking is unambiguous.
        self.embedder = FakeEmbedder({
            "passage about cats": [1.0, 0.0, 0.0],
            "passage about dogs": [0.9, 0.1, 0.0],
            "passage about the solar system": [0.0, 0.0, 1.0],
            "Tell me about cats": [1.0, 0.0, 0.0],
        })
        self.classifier = FakeNLIClassifier({})  # unused in these tests

    def test_retrieve_ranks_most_similar_first(self):
        verifier = EvidenceVerifier(
            self.kb, embedder=self.embedder, nli_classifier=self.classifier, top_k=3
        )
        results = verifier.retrieve("Tell me about cats")
        self.assertEqual(results[0].source_id, "KB-A")
        self.assertEqual(results[1].source_id, "KB-B")
        self.assertEqual(results[2].source_id, "KB-C")
        self.assertGreater(results[0].relevance_score, results[1].relevance_score)

    def test_retrieve_respects_top_k(self):
        verifier = EvidenceVerifier(
            self.kb, embedder=self.embedder, nli_classifier=self.classifier, top_k=2
        )
        results = verifier.retrieve("Tell me about cats")
        self.assertEqual(len(results), 2)


class TestVerifyClaim(unittest.TestCase):
    def setUp(self):
        self.kb = [{"source_id": "KB-001", "text": "The Earth orbits the Sun."}]
        self.embedder = FakeEmbedder({
            "The Earth orbits the Sun.": [1.0, 0.0],
            "The Earth goes around the Sun.": [0.95, 0.05],   # high relevance -> entailment
            "The Sun orbits the Earth.": [0.9, 0.1],           # high relevance -> contradiction
            "Bananas are a good source of potassium.": [0.0, 1.0],  # low relevance -> off-topic
        })

    def test_entailment_yields_supported_with_evidence(self):
        classifier = FakeNLIClassifier({
            ("The Earth orbits the Sun.", "The Earth goes around the Sun."): {
                "entailment": 0.93, "contradiction": 0.02, "neutral": 0.05,
            }
        })
        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=classifier)
        result = verifier.verify_claim("The Earth goes around the Sun.", claim_id=1)

        self.assertEqual(result.verdict, VERDICT_SUPPORTED)
        self.assertEqual(result.confidence, 0.93)
        self.assertEqual(result.claim_id, 1)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].source_id, "KB-001")

    def test_contradiction_yields_contradicted_with_evidence(self):
        classifier = FakeNLIClassifier({
            ("The Earth orbits the Sun.", "The Sun orbits the Earth."): {
                "entailment": 0.03, "contradiction": 0.9, "neutral": 0.07,
            }
        })
        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=classifier)
        result = verifier.verify_claim("The Sun orbits the Earth.")

        self.assertEqual(result.verdict, VERDICT_CONTRADICTED)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(len(result.evidence), 1)

    def test_no_relevant_passage_yields_unverifiable_without_calling_nli(self):
        class ExplodingClassifier:
            def __call__(self, premise, hypothesis):
                raise AssertionError("NLI should not be called when no passage is relevant")

        verifier = EvidenceVerifier(
            self.kb, embedder=self.embedder, nli_classifier=ExplodingClassifier(),
            relevance_threshold=0.35,
        )
        result = verifier.verify_claim("Bananas are a good source of potassium.")

        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(result.evidence, [])
        self.assertTrue(0.0 <= result.confidence <= 1.0)

    def test_neutral_nli_outcome_yields_unverifiable_with_no_evidence(self):
        classifier = FakeNLIClassifier({
            ("The Earth orbits the Sun.", "The Earth goes around the Sun."): {
                "entailment": 0.2, "contradiction": 0.1, "neutral": 0.7,
            }
        })
        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=classifier)
        result = verifier.verify_claim("The Earth goes around the Sun.")

        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.evidence, [])

    def test_confidence_always_in_valid_range(self):
        classifier = FakeNLIClassifier({
            ("The Earth orbits the Sun.", "The Earth goes around the Sun."): {
                "entailment": 0.93, "contradiction": 0.02, "neutral": 0.05,
            }
        })
        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=classifier)
        result = verifier.verify_claim("The Earth goes around the Sun.")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestVerifyClaimsBatchAndIntegrationWithPerson1(unittest.TestCase):
    def setUp(self):
        self.kb = [{"source_id": "KB-001", "text": "The Earth orbits the Sun."}]
        self.embedder = FakeEmbedder({
            "The Earth orbits the Sun.": [1.0, 0.0],
        })
        self.classifier = FakeNLIClassifier({
            ("The Earth orbits the Sun.", "The Earth orbits the Sun."): {
                "entailment": 0.99, "contradiction": 0.0, "neutral": 0.01,
            }
        })

    def test_accepts_plain_strings(self):
        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=self.classifier)
        results = verifier.verify_claims(["The Earth orbits the Sun."])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict, VERDICT_SUPPORTED)
        self.assertIsNone(results[0].claim_id)

    def test_accepts_person1_claim_objects(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from claim_extraction import extract_claims

        extraction = extract_claims("The Earth orbits the Sun.")
        self.assertEqual(extraction.status, "ok")

        verifier = EvidenceVerifier(self.kb, embedder=self.embedder, nli_classifier=self.classifier)
        results = verifier.verify_claims(extraction.claims)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].claim_id, extraction.claims[0].claim_id)
        self.assertEqual(results[0].claim_text, extraction.claims[0].text)
        self.assertEqual(results[0].verdict, VERDICT_SUPPORTED)


def _has_internet(host: str = "huggingface.co", timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def _real_models_available() -> bool:
    import importlib.util

    has_packages = (
        importlib.util.find_spec("sentence_transformers") is not None
        and importlib.util.find_spec("transformers") is not None
    )
    return has_packages and _has_internet()


@unittest.skipUnless(
    _real_models_available(),
    "sentence-transformers/transformers not installed or HuggingFace Hub unreachable "
    "(expected in network-restricted sandboxes; run this test on your own machine).",
)
class TestRealModelIntegration(unittest.TestCase):
    """End-to-end check with the actual models named in the SRS (REQ-5/REQ-6).

    This is intentionally the only test that downloads anything. It is
    skipped automatically wherever HuggingFace Hub isn't reachable, so it
    never blocks `pytest` in CI or in an offline sandbox — but you should
    run it at least once on your own machine before marking this module done.
    """

    def test_verifies_a_true_and_a_false_claim(self):
        kb = load_knowledge_base(KB_PATH)
        verifier = EvidenceVerifier(kb)

        supported = verifier.verify_claim("The Earth orbits the Sun.")
        self.assertEqual(supported.verdict, VERDICT_SUPPORTED)

        contradicted = verifier.verify_claim("The Sun orbits the Earth.")
        self.assertEqual(contradicted.verdict, VERDICT_CONTRADICTED)


if __name__ == "__main__":
    unittest.main()
