"""Task 1.1 (SRS REQ-1.2): assess the top 3 relevant passages and aggregate them.

Fake embedder/NLI models; no downloads, network or database.
Run:  python -m pytest tests/test_top3_evidence.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.auth import hash_api_key
from api.config import Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from evidence_verification import (
    DEFAULT_NLI_TOP_K,
    DEFAULT_TOP_K,
    EvidenceVerifier,
    PassageAssessment,
    VERDICT_CONTRADICTED,
    VERDICT_SUPPORTED,
    VERDICT_UNVERIFIABLE,
    aggregate_verdict,
)

CLAIM = "The Earth orbits the Sun."

ENTAIL = {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}
CONTRA = {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}
NEUTRAL = {"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8}


def unit(x):
    """2-D unit vector whose cosine with [1, 0] is x (relevance = x)."""
    return [x, (1 - x * x) ** 0.5]


class Fixture:
    """KB of passages P1..Pn with chosen relevance and NLI outcome per passage."""

    def __init__(self, passages):
        # passages: list of (source_id, relevance, nli_scores)
        self.kb = [{"source_id": sid, "text": f"passage {sid}"} for sid, _, _ in passages]
        self.vectors = {f"passage {sid}": unit(rel) for sid, rel, _ in passages}
        self.vectors[CLAIM] = [1.0, 0.0]
        self.nli = {f"passage {sid}": scores for sid, _, scores in passages}
        self.nli_calls = []

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 1.0]) for t in texts]

    def classify(self, premise, hypothesis):
        self.nli_calls.append(premise)
        return self.nli[premise]

    def verifier(self, **kwargs):
        return EvidenceVerifier(self.kb, embedder=self.embed, nli_classifier=self.classify, **kwargs)


def assessment(sid, relevance, scores):
    label = max(scores, key=scores.get)
    return PassageAssessment(sid, f"passage {sid}", relevance, label, scores[label], scores)


class TestDefaults(unittest.TestCase):
    def test_retrieves_five_assesses_three(self):
        self.assertEqual(DEFAULT_TOP_K, 5)
        self.assertEqual(DEFAULT_NLI_TOP_K, 3)

    def test_rejects_invalid_parameters(self):
        fx = Fixture([("P1", 0.9, ENTAIL)])
        with self.assertRaises(ValueError):
            fx.verifier(nli_top_k=0)
        with self.assertRaises(ValueError):
            fx.verifier(conflict_margin=1.5)


class TestThreeRelevantPassages(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture([
            ("P1", 0.90, ENTAIL),
            ("P2", 0.80, NEUTRAL),
            ("P3", 0.70, ENTAIL),
            ("P4", 0.60, CONTRA),  # 4th most relevant: retrieved but not assessed
            ("P5", 0.50, CONTRA),
        ])
        self.result = self.fx.verifier().verify_claim(CLAIM, claim_id=7)

    def test_only_top_three_are_assessed(self):
        self.assertEqual(sorted(self.fx.nli_calls), ["passage P1", "passage P2", "passage P3"])
        self.assertEqual([a.source_id for a in self.result.assessed_passages], ["P1", "P2", "P3"])

    def test_assessments_carry_id_text_relevance_and_nli(self):
        first = self.result.assessed_passages[0]
        self.assertEqual((first.source_id, first.text), ("P1", "passage P1"))
        self.assertAlmostEqual(first.relevance_score, 0.9, places=3)
        self.assertEqual(first.nli_label, "entailment")
        self.assertEqual(first.nli_confidence, 0.9)
        self.assertEqual(first.nli_scores, ENTAIL)

    def test_agreeing_entailments_are_supported_and_cited(self):
        self.assertEqual(self.result.verdict, VERDICT_SUPPORTED)
        self.assertEqual(self.result.confidence, 0.9)
        self.assertEqual([e.source_id for e in self.result.evidence], ["P1", "P3"])
        self.assertEqual(
            [a.supports_verdict for a in self.result.assessed_passages], [True, False, True]
        )


class TestFewerThanThree(unittest.TestCase):
    def test_single_passage_kb(self):
        fx = Fixture([("P1", 0.9, CONTRA)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_CONTRADICTED)
        self.assertEqual(len(result.assessed_passages), 1)
        self.assertEqual([e.source_id for e in result.evidence], ["P1"])

    def test_passages_below_threshold_are_not_assessed(self):
        fx = Fixture([("P1", 0.9, ENTAIL), ("P2", 0.5, ENTAIL), ("P3", 0.2, CONTRA)])
        result = fx.verifier(relevance_threshold=0.35).verify_claim(CLAIM)
        self.assertEqual([a.source_id for a in result.assessed_passages], ["P1", "P2"])
        self.assertNotIn("passage P3", fx.nli_calls)
        self.assertEqual(result.verdict, VERDICT_SUPPORTED)


class TestRelevanceThreshold(unittest.TestCase):
    def test_nothing_relevant_keeps_unverifiable_without_nli(self):
        fx = Fixture([("P1", 0.30, ENTAIL), ("P2", 0.20, CONTRA)])
        result = fx.verifier(relevance_threshold=0.35).verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(fx.nli_calls, [])
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.assessed_passages, [])
        self.assertEqual(result.confidence, 0.7)  # unchanged: 1 - best relevance

    def test_all_neutral_is_unverifiable_without_citations(self):
        fx = Fixture([("P1", 0.9, NEUTRAL), ("P2", 0.8, NEUTRAL)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(len(result.assessed_passages), 2)
        self.assertFalse(any(a.supports_verdict for a in result.assessed_passages))


class TestConflictingEvidence(unittest.TestCase):
    def test_relevant_contradiction_beats_tangential_entailment(self):
        fx = Fixture([("P1", 0.90, CONTRA), ("P2", 0.40, ENTAIL)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_CONTRADICTED)
        self.assertEqual([e.source_id for e in result.evidence], ["P1"])

    def test_relevant_entailment_beats_tangential_contradiction(self):
        # Real-model case: "Pacific is the largest ocean" vs a tangential passage.
        fx = Fixture([("P1", 0.81, ENTAIL), ("P2", 0.37, CONTRA)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_SUPPORTED)

    def test_cosine_rank_alone_does_not_decide(self):
        # Most relevant passage is weakly decisive; a slightly less relevant one
        # is strongly decisive the other way: NLI strength wins, not rank.
        weak_entail = {"entailment": 0.4, "contradiction": 0.3, "neutral": 0.3}
        strong_contra = {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01}
        fx = Fixture([("P1", 0.80, weak_entail), ("P2", 0.75, strong_contra)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_CONTRADICTED)
        self.assertEqual(result.confidence, 0.98)

    def test_evenly_matched_conflict_is_unverifiable(self):
        fx = Fixture([("P1", 0.80, ENTAIL), ("P2", 0.78, CONTRA)])
        result = fx.verifier().verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.assessed_passages), 2)
        self.assertTrue(0.0 <= result.confidence <= 1.0)

    def test_margin_zero_lets_the_stronger_side_win(self):
        fx = Fixture([("P1", 0.80, ENTAIL), ("P2", 0.78, CONTRA)])
        result = fx.verifier(conflict_margin=0.0).verify_claim(CLAIM)
        self.assertEqual(result.verdict, VERDICT_SUPPORTED)


class TestAggregateVerdict(unittest.TestCase):
    def test_requires_assessments(self):
        with self.assertRaises(ValueError):
            aggregate_verdict([])

    def test_supporting_passages_are_strongest_first(self):
        a = assessment("A", 0.5, ENTAIL)
        b = assessment("B", 0.9, ENTAIL)
        verdict, confidence, supporting = aggregate_verdict([a, b])
        self.assertEqual(verdict, VERDICT_SUPPORTED)
        self.assertEqual([s.source_id for s in supporting], ["B", "A"])

    def test_neutral_passages_do_not_dilute_a_verdict(self):
        verdict, _, supporting = aggregate_verdict(
            [assessment("N1", 0.95, NEUTRAL), assessment("C", 0.6, CONTRA)]
        )
        self.assertEqual(verdict, VERDICT_CONTRADICTED)
        self.assertEqual([s.source_id for s in supporting], ["C"])


class TestApiCompatibility(unittest.TestCase):
    def setUp(self):
        fx = Fixture([("P1", 0.90, ENTAIL), ("P2", 0.80, NEUTRAL), ("P3", 0.70, ENTAIL)])
        key = "k" * 32
        settings = Settings(api_key_hashes=(hash_api_key(key),), database_url="postgresql+psycopg://unused")

        class Logger:
            def record(self, entry):
                pass

            def purge_older_than(self, days):
                return 0

        app = create_app(settings, pipeline=VerificationPipeline(fx.verifier()), request_logger=Logger())
        with TestClient(app) as client:
            self.body = client.post("/verify", json={"text": CLAIM}, headers={"X-API-Key": key}).json()
        self.claim = self.body["claims"][0]

    def test_existing_fields_keep_their_shape(self):
        self.assertEqual(self.claim["verdict"], "Supported")
        self.assertEqual(self.claim["confidence"], 0.9)
        for citation in self.claim["citations"]:
            self.assertEqual(set(citation), {"source_id", "text", "relevance_score"})
        self.assertEqual([c["source_id"] for c in self.claim["citations"]], ["P1", "P3"])

    def test_evidence_passages_lists_all_three_assessments(self):
        passages = self.claim["evidence_passages"]
        self.assertEqual([p["source_id"] for p in passages], ["P1", "P2", "P3"])
        self.assertEqual([p["nli_label"] for p in passages], ["entailment", "neutral", "entailment"])
        self.assertEqual([p["supports_verdict"] for p in passages], [True, False, True])
        self.assertEqual(
            set(passages[0]),
            {"source_id", "text", "relevance_score", "nli_label", "nli_confidence", "nli_scores", "supports_verdict"},
        )
        self.assertEqual(set(passages[0]["nli_scores"]), {"entailment", "contradiction", "neutral"})


if __name__ == "__main__":
    unittest.main()
