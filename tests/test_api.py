
"""API tests for POST /verify (REQ-12 to REQ-15).

Uses fake embedder/NLI models (no downloads) and an in-memory request
logger (no database credentials). PostgreSQL-specific logging code is
covered in tests/test_request_log.py.
"""

import sys
import typing
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import evidence_verification
from api.auth import generate_api_key, hash_api_key, is_valid_api_key
from api.config import ConfigError, Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from api.schemas import Verdict
from evidence_verification import EvidenceVerifier


API_KEY = "A" * 16 + "b" * 16
OTHER_KEY = "Z" * 32

KB = [
    {
        "source_id": "KB-001",
        "text": "The Earth orbits the Sun once approximately every 365.25 days.",
    },
    {
        "source_id": "KB-002",
        "text": "Water is made of hydrogen and oxygen.",
    },
]


def fake_embedder(texts):
    """Keyword vectors: [astronomy, chemistry, other]."""
    vectors = []

    for t in texts:
        low = t.lower()

        vectors.append([
            1.0 if ("earth" in low or "sun" in low) else 0.0,
            1.0 if ("water" in low or "hydrogen" in low) else 0.0,
            0.0 if (
                "earth" in low
                or "sun" in low
                or "water" in low
                or "hydrogen" in low
            ) else 1.0,
        ])

    return vectors


def fake_nli(premise, hypothesis):
    if hypothesis.startswith("The Earth orbits"):
        return {
            "entailment": 0.95,
            "contradiction": 0.03,
            "neutral": 0.02,
        }

    if hypothesis.startswith("The Sun orbits"):
        return {
            "entailment": 0.04,
            "contradiction": 0.9,
            "neutral": 0.06,
        }

    return {
        "entailment": 0.1,
        "contradiction": 0.1,
        "neutral": 0.8,
    }


class FakeRequestLogger:
    def __init__(self, fail=False):
        self.entries = []
        self.purge_calls = []
        self.fail = fail

    def record(self, entry):
        if self.fail:
            raise RuntimeError(
                "db down: postgresql://user:secret@host/db"
            )
        self.entries.append(entry)

    def purge_older_than(self, days):
        self.purge_calls.append(days)
        return 0


class ExplodingPipeline:
    def run(self, text):
        raise RuntimeError("model crashed with internal detail")


def make_settings(**overrides):
    values = dict(
        api_key_hashes=(hash_api_key(API_KEY),),
        database_url="postgresql+psycopg://unused-in-tests",
        log_retention_days=14,
    )

    values.update(overrides)
    return Settings(**values)


def make_client(pipeline=None, request_logger=None, settings=None):
    if pipeline is None:
        verifier = EvidenceVerifier(
            KB,
            embedder=fake_embedder,
            nli_classifier=fake_nli,
        )
        pipeline = VerificationPipeline(verifier)

    request_logger = request_logger or FakeRequestLogger()

    app = create_app(
        settings or make_settings(),
        pipeline=pipeline,
        request_logger=request_logger,
    )

    return TestClient(app), request_logger


SAMPLE = (
    "The Earth orbits the Sun. "
    "The Sun orbits the Earth. "
    "Bananas are a good source of potassium."
)


class ApiTestCase(unittest.TestCase):

    def setUp(self):
        self.client, self.log = make_client()
        self.client.__enter__()
        self.addCleanup(
            self.client.__exit__,
            None,
            None,
            None,
        )

    def post(self, json=None, key=API_KEY, **kwargs):
        headers = kwargs.pop("headers", {})

        if key is not None:
            headers["X-API-Key"] = key

        return self.client.post(
            "/verify",
            json=json,
            headers=headers,
            **kwargs,
        )


class TestVerifySuccess(ApiTestCase):

    def test_returns_appendix_b_shape(self):
        res = self.post({"text": SAMPLE})

        self.assertEqual(res.status_code, 200)

        body = res.json()

        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["processing_time_ms"], int)

        self.assertEqual(
            [c["claim_id"] for c in body["claims"]],
            [1, 2, 3],
        )

        self.assertEqual(
            [c["verdict"] for c in body["claims"]],
            ["Supported", "Contradicted", "Unverifiable"],
        )

        first = body["claims"][0]

        self.assertEqual(
            set(first),
            {
                "claim_id",
                "text",
                "span",
                "source_sentence",
                "source_sentence_span",
                "verdict",
                "confidence",
                "citations",
            },
        )

        self.assertEqual(first["confidence"], 0.95)

        self.assertEqual(
            first["citations"],
            [
                {
                    "source_id": "KB-001",
                    "text": KB[0]["text"],
                    "relevance_score": 1.0,
                }
            ],
        )

    def test_spans_map_back_to_request_text(self):
        body = self.post({"text": SAMPLE}).json()

        for claim in body["claims"]:
            span = claim["span"]
            sent = claim["source_sentence_span"]

            self.assertEqual(
                SAMPLE[span["start_char"]:span["end_char"]],
                claim["text"],
            )

            self.assertEqual(
                SAMPLE[sent["start_char"]:sent["end_char"]],
                claim["source_sentence"],
            )

    def test_unverifiable_claim_has_no_citations(self):
        claims = self.post({"text": SAMPLE}).json()["claims"]

        self.assertEqual(claims[2]["citations"], [])

        for claim in claims:
            self.assertGreaterEqual(claim["confidence"], 0.0)
            self.assertLessEqual(claim["confidence"], 1.0)

    # Issue #13: Evidence citations for contradicted claims

    def test_contradicted_claim_includes_exact_evidence_citation(self):
        body = self.post({"text": SAMPLE}).json()
        claim = body["claims"][1]

        self.assertEqual(claim["verdict"], "Contradicted")
        self.assertEqual(claim["claim_id"], 2)

        self.assertEqual(
            claim["citations"],
            [
                {
                    "source_id": "KB-001",
                    "text": KB[0]["text"],
                    "relevance_score": 1.0,
                }
            ],
        )

    def test_citations_are_associated_with_correct_claims(self):
        body = self.post({"text": SAMPLE}).json()
        claims = body["claims"]

        self.assertEqual(claims[0]["claim_id"], 1)
        self.assertEqual(claims[0]["verdict"], "Supported")
        self.assertEqual(
            claims[0]["citations"][0]["source_id"],
            "KB-001",
        )

        self.assertEqual(claims[1]["claim_id"], 2)
        self.assertEqual(claims[1]["verdict"], "Contradicted")
        self.assertEqual(
            claims[1]["citations"][0]["source_id"],
            "KB-001",
        )

        self.assertEqual(claims[2]["claim_id"], 3)
        self.assertEqual(claims[2]["verdict"], "Unverifiable")
        self.assertEqual(claims[2]["citations"], [])

    def test_citation_fields_are_available_in_api_response(self):
        body = self.post({"text": SAMPLE}).json()

        for claim in body["claims"]:
            self.assertIn("citations", claim)

            for citation in claim["citations"]:
                self.assertEqual(
                    set(citation),
                    {
                        "source_id",
                        "text",
                        "relevance_score",
                    },
                )

    def test_no_verifiable_claims_is_200_with_message(self):
        res = self.post({
            "text": "I think this is wonderful. Is it?"
        })

        self.assertEqual(res.status_code, 200)

        body = res.json()

        self.assertEqual(
            body["status"],
            "no_verifiable_claims",
        )
        self.assertEqual(
            body["message"],
            "No verifiable claims found",
        )
        self.assertEqual(body["claims"], [])

    def test_whitespace_text_follows_extraction_module(self):
        res = self.post({"text": "   "})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json()["status"],
            "no_verifiable_claims",
        )

    def test_explicit_default_knowledge_base_id_accepted(self):
        res = self.post({
            "text": SAMPLE,
            "knowledge_base_id": "wikipedia_subset",
        })

        self.assertEqual(res.status_code, 200)

    def test_verdict_literal_matches_verification_module(self):
        self.assertEqual(
            set(typing.get_args(Verdict)),
            {
                evidence_verification.VERDICT_SUPPORTED,
                evidence_verification.VERDICT_CONTRADICTED,
                evidence_verification.VERDICT_UNVERIFIABLE,
            },
        )


class TestAuthentication(ApiTestCase):

    def test_missing_key_is_401(self):
        res = self.post({"text": SAMPLE}, key=None)

        self.assertEqual(res.status_code, 401)
        self.assertIn("X-API-Key", res.json()["detail"])

    def test_wrong_key_is_401(self):
        self.assertEqual(
            self.post({"text": SAMPLE}, key=OTHER_KEY).status_code,
            401,
        )

    def test_empty_key_is_401(self):
        self.assertEqual(
            self.post({"text": SAMPLE}, key="").status_code,
            401,
        )

    def test_auth_checked_before_body_validation(self):
        self.assertEqual(
            self.post({"wrong": 1}, key=None).status_code,
            401,
        )

    def test_multiple_configured_keys(self):
        client, _ = make_client(
            settings=make_settings(
                api_key_hashes=(
                    hash_api_key(OTHER_KEY),
                    hash_api_key(API_KEY),
                )
            )
        )

        with client:
            res = client.post(
                "/verify",
                json={"text": SAMPLE},
                headers={"X-API-Key": API_KEY},
            )

        self.assertEqual(res.status_code, 200)


class TestBadRequests(ApiTestCase):

    def assert_400(self, res, contains):
        self.assertEqual(res.status_code, 400, res.text)
        self.assertIsInstance(res.json()["detail"], str)
        self.assertIn(contains, res.json()["detail"])

    def test_missing_text(self):
        self.assert_400(self.post({}), "text")

    def test_non_string_text(self):
        self.assert_400(self.post({"text": 123}), "text")

    def test_invalid_json(self):
        res = self.post(
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(res.status_code, 400)

    def test_input_over_2000_words(self):
        self.assert_400(
            self.post({"text": "word " * 2001}),
            "maximum allowed is 2000",
        )

    def test_knowledge_base_id_too_long(self):
        self.assert_400(
            self.post({
                "text": SAMPLE,
                "knowledge_base_id": "k" * 65,
            }),
            "knowledge_base_id",
        )

    def test_unknown_knowledge_base_id(self):
        self.assert_400(
            self.post({
                "text": SAMPLE,
                "knowledge_base_id": "other_kb",
            }),
            "Unknown knowledge_base_id",
        )

    def test_openapi_documents_400_not_422(self):
        responses = (
            self.client.get("/openapi.json")
            .json()["paths"]["/verify"]["post"]["responses"]
        )

        self.assertEqual(
            set(responses),
            {"200", "400", "401", "500"},
        )


class TestServerError(unittest.TestCase):

    def test_pipeline_failure_is_500_without_internal_details(self):
        client, log = make_client(
            pipeline=ExplodingPipeline()
        )

        with client, self.assertLogs(
            "truthlayer.api",
            level="ERROR",
        ):
            res = client.post(
                "/verify",
                json={"text": SAMPLE},
                headers={"X-API-Key": API_KEY},
            )

        self.assertEqual(res.status_code, 500)
        self.assertNotIn("internal detail", res.text)
        self.assertEqual(
            log.entries[0].error_type,
            "pipeline_error",
        )


class TestRequestLogging(ApiTestCase):

    def test_success_logs_metadata_only(self):
        self.post({"text": SAMPLE})

        self.assertEqual(len(self.log.entries), 1)

        entry = self.log.entries[0]

        self.assertEqual(
            (entry.method, entry.path, entry.status_code),
            ("POST", "/verify", 200),
        )
        self.assertEqual(entry.input_char_count, len(SAMPLE))
        self.assertEqual(entry.input_word_count, 17)
        self.assertEqual(
            entry.knowledge_base_id,
            "wikipedia_subset",
        )

        self.assertEqual(
            (
                entry.claim_count,
                entry.supported_count,
                entry.contradicted_count,
                entry.unverifiable_count,
            ),
            (3, 1, 1, 1),
        )

        self.assertIsNone(entry.error_type)
        self.assertIsInstance(entry.processing_time_ms, int)

        for value in vars(entry).values():
            if isinstance(value, str):
                self.assertNotIn("Earth", value)

    def test_rejected_requests_are_logged(self):
        self.post({"text": SAMPLE}, key=None)
        self.post({"text": "word " * 2001})
        self.post({
            "text": SAMPLE,
            "knowledge_base_id": "other_kb",
        })
        self.post({})

        self.assertEqual(
            [
                (entry.status_code, entry.error_type)
                for entry in self.log.entries
            ],
            [
                (401, "unauthorized"),
                (400, "input_too_long"),
                (400, "unknown_knowledge_base"),
                (400, "invalid_request"),
            ],
        )

        self.assertIsNone(
            self.log.entries[2].knowledge_base_id
        )

    def test_logging_failure_does_not_fail_request(self):
        client, _ = make_client(
            request_logger=FakeRequestLogger(fail=True)
        )

        with client, self.assertLogs(
            "truthlayer.api",
            level="ERROR",
        ) as logs:
            res = client.post(
                "/verify",
                json={"text": SAMPLE},
                headers={"X-API-Key": API_KEY},
            )

        self.assertEqual(res.status_code, 200)
        self.assertNotIn(
            "secret",
            "\n".join(logs.output),
        )

    def test_retention_purge_runs_on_startup(self):
        self.assertEqual(self.log.purge_calls, [14])

    def test_cors_preflight_not_logged(self):
        self.client.options(
            "/verify",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(self.log.entries, [])


class TestCors(ApiTestCase):

    def test_preflight_allows_frontend_origin_and_api_key_header(self):
        res = self.client.options(
            "/verify",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-api-key",
            },
        )

        self.assertEqual(res.status_code, 200)

        self.assertEqual(
            res.headers["access-control-allow-origin"],
            "http://localhost:3000",
        )

        self.assertIn(
            "x-api-key",
            res.headers["access-control-allow-headers"].lower(),
        )

    def test_other_origin_rejected(self):
        res = self.client.options(
            "/verify",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertNotIn(
            "access-control-allow-origin",
            res.headers,
        )

    def test_error_responses_carry_cors_headers(self):
        res = self.post(
            {"text": SAMPLE},
            key=None,
            headers={"Origin": "http://localhost:3000"},
        )

        self.assertEqual(res.status_code, 401)

        self.assertEqual(
            res.headers["access-control-allow-origin"],
            "http://localhost:3000",
        )


class TestAuthHelpers(unittest.TestCase):

    def test_hash_is_sha256_hex(self):
        self.assertEqual(len(hash_api_key(API_KEY)), 64)
        self.assertNotEqual(hash_api_key(API_KEY), API_KEY)

    def test_is_valid_api_key(self):
        hashes = (hash_api_key(API_KEY),)

        self.assertTrue(is_valid_api_key(API_KEY, hashes))
        self.assertFalse(is_valid_api_key(OTHER_KEY, hashes))
        self.assertFalse(is_valid_api_key(None, hashes))
        self.assertFalse(is_valid_api_key(API_KEY, ()))

    def test_generated_key_is_32_alphanumeric(self):
        key = generate_api_key()

        self.assertEqual(len(key), 32)
        self.assertTrue(key.isalnum() and key.isascii())


class TestSettings(unittest.TestCase):

    BASE = {
        "TRUTHLAYER_API_KEY_HASHES": hash_api_key(API_KEY),
        "TRUTHLAYER_DATABASE_URL": (
            "postgresql+psycopg://u:topsecret@localhost/db"
        ),
    }

    def test_defaults(self):
        settings = Settings.from_env(self.BASE)

        self.assertEqual(
            settings.knowledge_base_id,
            "wikipedia_subset",
        )
        self.assertEqual(
            settings.cors_origins,
            ("http://localhost:3000",),
        )
        self.assertEqual(settings.log_retention_days, 30)
        self.assertTrue(settings.knowledge_base_path.is_file())

    def test_missing_key_hashes_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({
                **self.BASE,
                "TRUTHLAYER_API_KEY_HASHES": "",
            })

    def test_plaintext_key_instead_of_hash_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({
                **self.BASE,
                "TRUTHLAYER_API_KEY_HASHES": API_KEY,
            })

    def test_missing_database_url_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({
                **self.BASE,
                "TRUTHLAYER_DATABASE_URL": "",
            })

    def test_invalid_retention_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({
                **self.BASE,
                "TRUTHLAYER_LOG_RETENTION_DAYS": "0",
            })

    def test_repr_hides_database_url(self):
        settings = Settings.from_env(self.BASE)

        self.assertNotIn("topsecret", repr(settings))


if __name__ == "__main__":
    unittest.main()
