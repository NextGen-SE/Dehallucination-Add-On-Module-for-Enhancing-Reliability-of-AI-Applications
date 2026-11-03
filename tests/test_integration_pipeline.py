"""Integration tests: claim extraction -> evidence retrieval -> NLI -> API (Issue #24).

Uses the REAL claim extractor, REAL EvidenceVerifier / VerificationPipeline and
the REAL FastAPI app. Only the two ML models (embedder, NLI) and the request
logger are replaced with small deterministic fakes, so no model download and no
database are needed.

Run:  python -m pytest tests/test_integration_pipeline.py -v
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
from claim_extraction import extract_claims
from evidence_verification import (
    VERDICT_CONTRADICTED,
    VERDICT_SUPPORTED,
    VERDICT_UNVERIFIABLE,
    EvidenceVerifier,
)

API_KEY = "I" * 16 + "t" * 16

KB = [
    {"source_id": "KB-001", "text": "The Earth orbits the Sun once approximately every 365.25 days."},
    {"source_id": "KB-002", "text": "Water is made of hydrogen and oxygen."},
]


# --------------------------------------------------------------------------- #
# Fakes (models + logger only)
# --------------------------------------------------------------------------- #

def fake_embedder(texts):
    """Keyword vectors: [astronomy, chemistry, other]."""
    out = []
    for t in texts:
        low = t.lower()
        astro = 1.0 if ("earth" in low or "sun" in low) else 0.0
        chem = 1.0 if ("water" in low or "hydrogen" in low) else 0.0
        out.append([astro, chem, 0.0 if (astro or chem) else 1.0])
    return out


class CountingNLI:
    """Rule-based NLI that also counts calls (to prove Unverifiable skips NLI)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, premise, hypothesis):
        self.calls += 1
        h = hypothesis.lower()
        if "sun orbits the earth" in h or "100 days" in h:
            return {"entailment": 0.04, "contradiction": 0.90, "neutral": 0.06}
        if "orbits the sun" in h or "hydrogen" in h:
            return {"entailment": 0.95, "contradiction": 0.03, "neutral": 0.02}
        return {"entailment": 0.10, "contradiction": 0.10, "neutral": 0.80}


class MemoryLogger:
    def __init__(self):
        self.entries = []

    def record(self, entry):
        self.entries.append(entry)

    def purge_older_than(self, days):
        return 0


class RaisingVerifier:
    """Stands in for a crashed model."""

    def verify_claims(self, claims):
        raise RuntimeError("model exploded: internal detail")


def make_settings():
    return Settings(
        api_key_hashes=(hash_api_key(API_KEY),),
        database_url="postgresql+psycopg://unused-in-tests",
    )


def make_verifier(nli=None):
    return EvidenceVerifier(KB, embedder=fake_embedder, nli_classifier=nli or CountingNLI())


# --------------------------------------------------------------------------- #
# Module-to-module: extractor -> verifier -> pipeline
# --------------------------------------------------------------------------- #

class TestExtractorToVerifier(unittest.TestCase):
    def test_claim_ids_carry_through_to_verification_results(self):
        extraction = extract_claims("The Earth orbits the Sun. The Sun orbits the Earth.")
        results = make_verifier().verify_claims(extraction.claims)
        self.assertEqual([c.claim_id for c in extraction.claims], [r.claim_id for r in results])

    def test_supported_and_contradicted_verdicts_with_evidence(self):
        extraction = extract_claims("The Earth orbits the Sun. The Sun orbits the Earth.")
        supported, contradicted = make_verifier().verify_claims(extraction.claims)

        self.assertEqual(supported.verdict, VERDICT_SUPPORTED)
        self.assertEqual(supported.evidence[0].source_id, "KB-001")
        self.assertEqual(contradicted.verdict, VERDICT_CONTRADICTED)
        self.assertEqual(contradicted.evidence[0].source_id, "KB-001")

    def test_unrelated_claim_is_unverifiable_and_skips_nli(self):
        nli = CountingNLI()
        extraction = extract_claims("Bananas are a good source of potassium.")
        (result,) = make_verifier(nli).verify_claims(extraction.claims)

        self.assertEqual(result.verdict, VERDICT_UNVERIFIABLE)
        self.assertEqual(result.evidence, [])
        self.assertEqual(nli.calls, 0)

    def test_pipeline_merges_extraction_and_verification(self):
        text = "The Earth orbits the Sun. Water contains hydrogen and oxygen."
        status, message, claims = VerificationPipeline(make_verifier()).run(text)

        self.assertEqual(status, "ok")
        self.assertIsNone(message)
        self.assertEqual(len(claims), 2)
        for claim in claims:
            self.assertEqual(text[claim.span.start_char:claim.span.end_char], claim.text)
            self.assertEqual(claim.verdict, VERDICT_SUPPORTED)
            self.assertTrue(claim.citations)

    def test_pipeline_no_claims_returns_explicit_status(self):
        status, message, claims = VerificationPipeline(make_verifier()).run("I think pizza is great!")
        self.assertEqual(status, "no_verifiable_claims")
        self.assertEqual(message, "No verifiable claims found")
        self.assertEqual(claims, [])


# --------------------------------------------------------------------------- #
# Full stack through HTTP
# --------------------------------------------------------------------------- #

class ApiIntegrationBase(unittest.TestCase):
    pipeline = None

    def setUp(self):
        self.logger = MemoryLogger()
        pipeline = self.pipeline or VerificationPipeline(make_verifier())
        app = create_app(make_settings(), pipeline=pipeline, request_logger=self.logger)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def post(self, body, key=API_KEY):
        headers = {"X-API-Key": key} if key else {}
        return self.client.post("/verify", json=body, headers=headers)


class TestVerifyEndpointEndToEnd(ApiIntegrationBase):
    TEXT = (
        "The Earth orbits the Sun. "
        "The Sun orbits the Earth. "
        "Bananas are a good source of potassium."
    )

    def test_full_response_for_mixed_text(self):
        resp = self.post({"text": self.TEXT})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["processing_time_ms"], int)
        verdicts = [c["verdict"] for c in body["claims"]]
        self.assertEqual(verdicts, ["Supported", "Contradicted", "Unverifiable"])

        for claim in body["claims"]:
            self.assertGreaterEqual(claim["confidence"], 0.0)
            self.assertLessEqual(claim["confidence"], 1.0)
            span = claim["span"]
            self.assertEqual(self.TEXT[span["start_char"]:span["end_char"]], claim["text"])

    def test_citations_match_verdicts(self):
        claims = self.post({"text": self.TEXT}).json()["claims"]
        supported, contradicted, unverifiable = claims

        self.assertEqual(supported["citations"][0]["source_id"], "KB-001")
        self.assertIn("365.25", supported["citations"][0]["text"])
        self.assertEqual(contradicted["citations"][0]["source_id"], "KB-001")
        self.assertEqual(unverifiable["citations"], [])

    def test_unverifiable_is_never_returned_as_supported(self):
        """NFR-4: insufficient evidence must default to Unverifiable."""
        claims = self.post({"text": "Bananas are a good source of potassium."}).json()["claims"]
        self.assertEqual(claims[0]["verdict"], "Unverifiable")

    def test_no_verifiable_claims_is_200_not_error(self):
        resp = self.post({"text": "I think pizza is great!"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "no_verifiable_claims")
        self.assertEqual(body["claims"], [])

    def test_successful_request_is_logged_without_text(self):
        self.post({"text": self.TEXT})
        self.assertEqual(len(self.logger.entries), 1)
        entry = self.logger.entries[0]
        self.assertEqual(entry.status_code, 200)
        self.assertNotIn("Earth", repr(entry))


class TestFailureCases(ApiIntegrationBase):
    def test_missing_key_is_401(self):
        self.assertEqual(self.post({"text": "The Earth orbits the Sun."}, key=None).status_code, 401)

    def test_wrong_key_is_401(self):
        self.assertEqual(self.post({"text": "The Earth orbits the Sun."}, key="x" * 32).status_code, 401)

    def test_missing_text_is_400(self):
        resp = self.post({})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("detail", resp.json())

    def test_non_string_text_is_400(self):
        self.assertEqual(self.post({"text": 12345}).status_code, 400)

    def test_over_2000_words_is_400(self):
        resp = self.post({"text": "word " * 2001})
        self.assertEqual(resp.status_code, 400)

    def test_failed_requests_are_logged(self):
        self.post({}, key=None)
        self.post({})
        self.assertEqual([e.status_code for e in self.logger.entries], [401, 400])

    def test_server_still_works_after_bad_requests(self):
        self.post({})
        self.post({"text": 1})
        self.assertEqual(self.post({"text": "The Earth orbits the Sun."}).status_code, 200)


class TestPipelineFailure(ApiIntegrationBase):
    pipeline = VerificationPipeline(RaisingVerifier())

    def test_verifier_crash_is_500_without_internal_details(self):
        resp = self.post({"text": "The Earth orbits the Sun."})
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn("exploded", resp.text)


if __name__ == "__main__":
    unittest.main()