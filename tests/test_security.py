"""Security tests for Issue #27 (SRS NFR-4, NFR-6, NFR-7, REQ-13, REQ-15).

Covers API-key authentication, invalid-input handling that must not crash the
server, and that submitted text never reaches logs or the request-log table.
Uses fake models, so no downloads or database credentials are needed.

Run:  python -m pytest tests/test_security.py -v
"""
import json
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from api.auth import hash_api_key, is_valid_api_key
from api.config import Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from api.request_log import SqlRequestLogger
from evidence_verification import EvidenceVerifier

API_KEY = "S" * 16 + "e" * 16
SENTINEL = "ZEBRA-SECRET-123"

KB = [{"source_id": "KB-001", "text": "The Earth orbits the Sun once approximately every 365.25 days."}]


def embedder(texts):
    return [[1.0, 0.0] if ("earth" in t.lower() or "sun" in t.lower()) else [0.0, 1.0] for t in texts]


def nli(premise, hypothesis):
    return {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}


class ExplodingPipeline:
    def run(self, text):
        raise RuntimeError("internal failure")


def make_client(pipeline=None, request_logger=None):
    settings = Settings(
        api_key_hashes=(hash_api_key(API_KEY),),
        database_url="postgresql+psycopg://user:dbpassword@unused/db",
    )
    pipeline = pipeline or VerificationPipeline(EvidenceVerifier(KB, embedder=embedder, nli_classifier=nli))
    app = create_app(settings, pipeline=pipeline, request_logger=request_logger)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client


class SqliteLogger(SqlRequestLogger):
    """Real SQL logger on in-memory SQLite, with a helper to dump every stored value."""

    def __init__(self):
        super().__init__(create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))
        self.create_schema()

    def dump(self) -> str:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM request_logs")).fetchall()
        return json.dumps([[str(v) for v in row] for row in rows])


class TestAuthentication(unittest.TestCase):
    def setUp(self):
        self.client = make_client(request_logger=SqliteLogger())
        self.addCleanup(self.client.__exit__, None, None, None)

    def post(self, key=None, body=None):
        headers = {"X-API-Key": key} if key is not None else {}
        return self.client.post("/verify", json=body or {"text": "The Earth orbits the Sun."}, headers=headers)

    def test_missing_key_is_401(self):
        self.assertEqual(self.post().status_code, 401)

    def test_empty_key_is_401(self):
        self.assertEqual(self.post(key="").status_code, 401)

    def test_wrong_key_is_401(self):
        self.assertEqual(self.post(key="x" * 32).status_code, 401)

    def test_key_with_wrong_case_is_401(self):
        self.assertEqual(self.post(key=API_KEY.swapcase()).status_code, 401)

    def test_key_with_extra_characters_is_401(self):
        self.assertEqual(self.post(key=API_KEY + "x").status_code, 401)

    def test_sql_injection_style_key_is_401(self):
        self.assertEqual(self.post(key="' OR '1'='1").status_code, 401)

    def test_valid_key_is_200(self):
        self.assertEqual(self.post(key=API_KEY).status_code, 200)

    def test_401_does_not_reveal_valid_key_or_hash(self):
        body = self.post(key="x" * 32).text
        self.assertNotIn(API_KEY, body)
        self.assertNotIn(hash_api_key(API_KEY), body)

    def test_auth_is_checked_before_body_validation(self):
        resp = self.client.post("/verify", json={"text": 5})
        self.assertEqual(resp.status_code, 401)

    def test_keys_are_configured_as_hashes_not_plaintext(self):
        settings = self.client.app.state.settings
        self.assertNotIn(API_KEY, settings.api_key_hashes)
        self.assertIn(hash_api_key(API_KEY), settings.api_key_hashes)
        self.assertNotIn(API_KEY, repr(settings))
        self.assertTrue(is_valid_api_key(API_KEY, settings.api_key_hashes))

    def test_database_password_not_in_settings_repr(self):
        self.assertNotIn("dbpassword", repr(self.client.app.state.settings))


class TestInvalidInputDoesNotCrash(unittest.TestCase):
    def setUp(self):
        self.client = make_client(request_logger=SqliteLogger())
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-API-Key": API_KEY}

    def assert_still_healthy(self):
        resp = self.client.post("/verify", json={"text": "The Earth orbits the Sun."}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)

    def test_bad_payload_sweep_returns_400_and_server_survives(self):
        bad_payloads = [
            {},
            {"text": None},
            {"text": 123},
            {"text": ["a", "b"]},
            {"text": {"nested": "x"}},
            {"text": "word " * 2001},
            {"text": "ok", "knowledge_base_id": "x" * 65},
            {"text": "ok", "knowledge_base_id": "does_not_exist"},
            {"text": "ok", "knowledge_base_id": 5},
            [],
            "just a string",
        ]
        for payload in bad_payloads:
            with self.subTest(payload=str(payload)[:40]):
                resp = self.client.post("/verify", json=payload, headers=self.headers)
                self.assertEqual(resp.status_code, 400)
                self.assertIn("detail", resp.json())
        self.assert_still_healthy()

    def test_invalid_json_and_wrong_content_type_return_400(self):
        for content in (b"{not json", b"", b"\xff\xfe\x00"):
            with self.subTest(content=content):
                resp = self.client.post(
                    "/verify", content=content, headers={**self.headers, "Content-Type": "application/json"}
                )
                self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/verify", content=b"text=hello", headers={**self.headers, "Content-Type": "text/plain"})
        self.assertEqual(resp.status_code, 400)
        self.assert_still_healthy()

    def test_odd_but_valid_text_is_handled_without_server_error(self):
        for odd in ["", "   ", "\n\n\t", "!!!???", "🙂" * 50, "<script>alert(1)</script>", "'; DROP TABLE request_logs;--"]:
            with self.subTest(text=odd[:20]):
                resp = self.client.post("/verify", json={"text": odd}, headers=self.headers)
                self.assertLess(resp.status_code, 500)
        self.assert_still_healthy()

    def test_unsupported_methods_are_rejected(self):
        self.assertEqual(self.client.get("/verify").status_code, 405)


class TestSensitiveDataIsNotLogged(unittest.TestCase):
    def setUp(self):
        self.db = SqliteLogger()
        self.client = make_client(request_logger=self.db)
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-API-Key": API_KEY}
        self.text = f"The Earth orbits the Sun. {SENTINEL} is a secret project code name."

    def test_submitted_text_not_in_logs_or_database_on_success(self):
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("truthlayer.test").debug("capture-start")
            resp = self.client.post("/verify", json={"text": self.text}, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(SENTINEL, "\n".join(captured.output))
        self.assertNotIn(SENTINEL, self.db.dump())

    def test_submitted_text_not_in_logs_or_database_on_validation_error(self):
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("truthlayer.test").debug("capture-start")
            resp = self.client.post("/verify", json={"text": self.text, "knowledge_base_id": "nope"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn(SENTINEL, "\n".join(captured.output))
        self.assertNotIn(SENTINEL, self.db.dump())

    def test_api_key_never_logged(self):
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("truthlayer.test").debug("capture-start")
            self.client.post("/verify", json={"text": self.text}, headers=self.headers)
            self.client.post("/verify", json={"text": self.text}, headers={"X-API-Key": "bad" * 10})
        self.assertNotIn(API_KEY, "\n".join(captured.output))
        self.assertNotIn(API_KEY, self.db.dump())
        self.assertNotIn("bad" * 10, self.db.dump())

    def test_database_rows_contain_only_metadata_columns(self):
        self.client.post("/verify", json={"text": self.text}, headers=self.headers)
        with self.db._engine.connect() as conn:
            columns = set(conn.execute(text("SELECT * FROM request_logs")).keys())
        self.assertEqual(
            columns,
            {"id", "created_at", "method", "path", "status_code", "processing_time_ms", "input_word_count",
             "input_char_count", "knowledge_base_id", "claim_count", "supported_count", "contradicted_count",
             "unverifiable_count", "error_type"},
        )


class TestServerErrorsDoNotLeak(unittest.TestCase):
    def test_500_hides_internal_details_and_text(self):
        db = SqliteLogger()
        client = make_client(pipeline=ExplodingPipeline(), request_logger=db)
        self.addCleanup(client.__exit__, None, None, None)
        resp = client.post("/verify", json={"text": f"The Earth orbits the Sun. {SENTINEL}"}, headers={"X-API-Key": API_KEY})
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn("internal failure", resp.text)
        self.assertNotIn(SENTINEL, resp.text)
        self.assertNotIn(SENTINEL, db.dump())


class TestNfr4Safety(unittest.TestCase):
    def test_unrelated_claim_is_never_supported(self):
        """NFR-4: with no relevant evidence the verdict must default to Unverifiable."""
        client = make_client(request_logger=SqliteLogger())
        self.addCleanup(client.__exit__, None, None, None)
        resp = client.post("/verify", json={"text": "Bananas are a good source of potassium."},
                           headers={"X-API-Key": API_KEY})
        verdicts = [c["verdict"] for c in resp.json()["claims"]]
        self.assertEqual(verdicts, ["Unverifiable"])


if __name__ == "__main__":
    unittest.main()