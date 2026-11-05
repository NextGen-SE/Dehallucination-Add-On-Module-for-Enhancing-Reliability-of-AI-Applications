"""Issue #26 end-to-end API workflow tests.

These tests exercise the real FastAPI route, claim extractor, evidence
retriever/NLI decision logic (with deterministic fake model adapters),
response schema, authentication, and request logging. They do not download
ML models or require a live PostgreSQL instance.
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.auth import hash_api_key
from api.config import Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from evidence_verification import EvidenceVerifier

API_KEY = "A" * 32
SAMPLE_TEXT = (
    "The Earth orbits the Sun. "
    "The Sun orbits the Earth. "
    "Bananas are a good source of potassium."
)
KNOWLEDGE_BASE = [
    {
        "source_id": "KB-001",
        "text": "The Earth orbits the Sun once approximately every 365.25 days.",
    },
    {
        "source_id": "KB-002",
        "text": "Water is made of hydrogen and oxygen.",
    },
]


def fake_embedder(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lower = text.lower()
        astronomy = "earth" in lower or "sun" in lower
        chemistry = "water" in lower or "hydrogen" in lower
        vectors.append([
            1.0 if astronomy else 0.0,
            1.0 if chemistry else 0.0,
            0.0 if astronomy or chemistry else 1.0,
        ])
    return vectors


def fake_nli(premise: str, hypothesis: str) -> dict[str, float]:
    if hypothesis.startswith("The Earth orbits"):
        return {"entailment": 0.95, "contradiction": 0.03, "neutral": 0.02}
    if hypothesis.startswith("The Sun orbits"):
        return {"entailment": 0.04, "contradiction": 0.90, "neutral": 0.06}
    return {"entailment": 0.10, "contradiction": 0.10, "neutral": 0.80}


class InMemoryRequestLogger:
    def __init__(self) -> None:
        self.entries = []

    def record(self, entry) -> None:
        self.entries.append(entry)

    def purge_older_than(self, days: int) -> int:
        return 0


class TestIssue26EndToEndWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        verifier = EvidenceVerifier(
            KNOWLEDGE_BASE,
            embedder=fake_embedder,
            nli_classifier=fake_nli,
        )
        pipeline = VerificationPipeline(verifier)
        self.request_logger = InMemoryRequestLogger()
        settings = Settings(
            api_key_hashes=(hash_api_key(API_KEY),),
            database_url="postgresql+psycopg://unused-in-e2e-tests",
            log_retention_days=30,
        )
        self.client_context = TestClient(
            create_app(settings, pipeline=pipeline, request_logger=self.request_logger)
        )
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def post_verify(self, payload, api_key: str | None = API_KEY):
        headers = {"X-API-Key": api_key} if api_key is not None else {}
        return self.client.post("/verify", json=payload, headers=headers)

    def test_dashboard_workflow_api_returns_all_verdicts_and_citations(self):
        response = self.post_verify({"text": SAMPLE_TEXT})
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(
            [claim["verdict"] for claim in body["claims"]],
            ["Supported", "Contradicted", "Unverifiable"],
        )
        self.assertIsInstance(body["processing_time_ms"], int)

        supported, contradicted, unverifiable = body["claims"]
        self.assertEqual(supported["citations"][0]["source_id"], "KB-001")
        self.assertEqual(contradicted["citations"][0]["source_id"], "KB-001")
        self.assertEqual(unverifiable["citations"], [])

        # Confirm claim and source sentence offsets map to the submitted text.
        for claim in body["claims"]:
            span = claim["span"]
            sentence_span = claim["source_sentence_span"]
            self.assertEqual(
                SAMPLE_TEXT[span["start_char"]:span["end_char"]], claim["text"]
            )
            self.assertEqual(
                SAMPLE_TEXT[sentence_span["start_char"]:sentence_span["end_char"]],
                claim["source_sentence"],
            )

    def test_missing_evidence_is_unverifiable_and_has_no_citation(self):
        response = self.post_verify({"text": "Bananas are a good source of potassium."})
        self.assertEqual(response.status_code, 200)
        claim = response.json()["claims"][0]
        self.assertEqual(claim["verdict"], "Unverifiable")
        self.assertEqual(claim["citations"], [])

    def test_invalid_input_and_authentication_return_clear_http_errors(self):
        invalid = self.post_verify({})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("detail", invalid.json())

        unauthenticated = self.post_verify({"text": SAMPLE_TEXT}, api_key=None)
        self.assertEqual(unauthenticated.status_code, 401)
        self.assertIn("X-API-Key", unauthenticated.json()["detail"])

    def test_no_checkable_claims_returns_successful_empty_result(self):
        response = self.post_verify({"text": "I think this is wonderful."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "no_verifiable_claims")
        self.assertEqual(response.json()["message"], "No verifiable claims found")
        self.assertEqual(response.json()["claims"], [])

    def test_request_metadata_is_logged_without_submitted_text(self):
        response = self.post_verify({"text": SAMPLE_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.request_logger.entries), 1)
        entry = self.request_logger.entries[0]
        self.assertEqual(entry.path, "/verify")
        self.assertEqual(entry.status_code, 200)
        self.assertGreater(entry.input_word_count, 0)
        self.assertFalse(hasattr(entry, "text"))


if __name__ == "__main__":
    unittest.main()
