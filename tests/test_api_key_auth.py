"""API key authentication tests for Issue #16 (SRS REQ-13, NFR-6).

Complements the basic auth checks in tests/test_api.py and
tests/test_security.py with: malformed headers, multiple keys loaded from the
environment, fail-closed configuration, and plaintext keys never being stored
or logged. Keys are generated at runtime; no real secret is in this file.
Uses fake models and in-memory SQLite, so no downloads or credentials needed.

Run:  python -m pytest tests/test_api_key_auth.py -v
"""
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from api.auth import API_KEY_HEADER, generate_api_key, hash_api_key
from api.config import ConfigError, Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from api.request_log import SqlRequestLogger
from evidence_verification import EvidenceVerifier

KB = [{"source_id": "KB-001", "text": "The Earth orbits the Sun once approximately every 365.25 days."}]
BODY = {"text": "The Earth orbits the Sun."}
DB_URL = "postgresql+psycopg://user:unused@localhost/db"


def embedder(texts):
    return [[1.0, 0.0] if ("earth" in t.lower() or "sun" in t.lower()) else [0.0, 1.0] for t in texts]


def nli(premise, hypothesis):
    return {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}


class CountingPipeline(VerificationPipeline):
    """Real pipeline with fake models that records how often it ran."""

    def __init__(self):
        super().__init__(EvidenceVerifier(KB, embedder=embedder, nli_classifier=nli))
        self.calls = 0

    def run(self, text):
        self.calls += 1
        return super().run(text)


class SqliteLogger(SqlRequestLogger):
    """Real SQL request logger on in-memory SQLite, able to dump all stored values."""

    def __init__(self):
        super().__init__(create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))
        self.create_schema()

    def dump(self) -> str:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM request_logs")).fetchall()
        return json.dumps([[str(v) for v in row] for row in rows])


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.key = generate_api_key()
        self.pipeline = CountingPipeline()
        self.db = SqliteLogger()
        self.client = self.make_client((hash_api_key(self.key),))

    def make_client(self, hashes):
        settings = Settings(api_key_hashes=hashes, database_url=DB_URL)
        app = create_app(settings, pipeline=self.pipeline, request_logger=self.db)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def post(self, headers):
        return self.client.post("/verify", json=BODY, headers=headers)

    def assert_rejected(self, headers):
        resp = self.post(headers)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json(), {"detail": f"Provide a valid key in the {API_KEY_HEADER} header."})
        return resp


class TestValidAndMissingKeys(AuthTestCase):
    def test_valid_key_allows_request(self):
        resp = self.post({API_KEY_HEADER: self.key})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.pipeline.calls, 1)

    def test_header_name_is_case_insensitive(self):
        self.assertEqual(self.post({API_KEY_HEADER.lower(): self.key}).status_code, 200)

    def test_missing_key_is_401(self):
        self.assert_rejected({})

    def test_empty_key_is_401(self):
        self.assert_rejected({API_KEY_HEADER: ""})

    def test_invalid_key_is_401(self):
        self.assert_rejected({API_KEY_HEADER: generate_api_key()})

    def test_rejected_request_never_reaches_pipeline(self):
        self.assert_rejected({})
        self.assert_rejected({API_KEY_HEADER: generate_api_key()})
        self.assertEqual(self.pipeline.calls, 0)

    def test_missing_and_invalid_keys_get_identical_responses(self):
        # Same status and body, so callers cannot tell which check failed.
        missing = self.assert_rejected({})
        invalid = self.assert_rejected({API_KEY_HEADER: generate_api_key()})
        self.assertEqual(missing.content, invalid.content)


class TestMalformedHeaders(AuthTestCase):
    def test_key_in_authorization_bearer_header_is_401(self):
        self.assert_rejected({"Authorization": f"Bearer {self.key}"})

    def test_bearer_prefix_inside_api_key_header_is_401(self):
        self.assert_rejected({API_KEY_HEADER: f"Bearer {self.key}"})

    def test_wrong_header_name_is_401(self):
        self.assert_rejected({"X-Api-Token": self.key})

    def test_whitespace_only_key_is_401(self):
        self.assert_rejected({API_KEY_HEADER: "   "})

    def test_non_ascii_key_is_401_not_500(self):
        self.assert_rejected({API_KEY_HEADER: "kéyÿ".encode("latin-1")})

    def test_oversized_key_is_401_not_500(self):
        self.assert_rejected({API_KEY_HEADER: "a" * 8000})

    def test_truncated_key_is_401(self):
        self.assert_rejected({API_KEY_HEADER: self.key[:-1]})

    def test_stored_hash_is_not_usable_as_a_key(self):
        # A leaked hash from config must not work as a credential.
        self.assert_rejected({API_KEY_HEADER: hash_api_key(self.key)})


class TestMultipleKeys(AuthTestCase):
    def test_every_configured_key_is_accepted_and_others_rejected(self):
        second = generate_api_key()
        client = self.make_client((hash_api_key(self.key), hash_api_key(second)))
        for key in (self.key, second):
            self.assertEqual(client.post("/verify", json=BODY, headers={API_KEY_HEADER: key}).status_code, 200)
        resp = client.post("/verify", json=BODY, headers={API_KEY_HEADER: generate_api_key()})
        self.assertEqual(resp.status_code, 401)

    def test_comma_separated_hashes_from_env(self):
        keys = [generate_api_key() for _ in range(3)]
        env = {
            "TRUTHLAYER_API_KEY_HASHES": " , ".join(hash_api_key(k).upper() for k in keys) + ",",
            "TRUTHLAYER_DATABASE_URL": DB_URL,
        }
        settings = Settings.from_env(env)
        self.assertEqual(settings.api_key_hashes, tuple(hash_api_key(k) for k in keys))


class TestConfiguration(unittest.TestCase):
    def env(self, hashes):
        return {"TRUTHLAYER_API_KEY_HASHES": hashes, "TRUTHLAYER_DATABASE_URL": DB_URL}

    def test_unset_hashes_variable_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({"TRUTHLAYER_DATABASE_URL": DB_URL})

    def test_separator_only_hashes_fail(self):
        with self.assertRaises(ConfigError):
            Settings.from_env(self.env(" , ,"))

    def test_one_invalid_entry_among_valid_hashes_fails(self):
        with self.assertRaises(ConfigError):
            Settings.from_env(self.env(f"{hash_api_key(generate_api_key())},not-a-hash"))

    def test_config_error_does_not_echo_plaintext_key(self):
        key = generate_api_key()
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env(self.env(key))
        self.assertNotIn(key, str(ctx.exception))

    def test_create_app_fails_closed_without_environment(self):
        # The API must refuse to start rather than run unauthenticated.
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigError):
            create_app()


class TestPlaintextKeyNeverStoredOrLogged(AuthTestCase):
    def test_settings_hold_only_hashes(self):
        settings = self.client.app.state.settings
        self.assertEqual(settings.api_key_hashes, (hash_api_key(self.key),))
        self.assertNotIn(self.key, repr(settings))
        self.assertNotIn(self.key, repr(vars(settings)))

    def test_valid_and_rejected_keys_not_in_logs_or_database(self):
        wrong_key = generate_api_key()
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("truthlayer.test").debug("capture-start")
            self.assertEqual(self.post({API_KEY_HEADER: self.key}).status_code, 200)
            self.assert_rejected({API_KEY_HEADER: wrong_key})
            self.assert_rejected({"Authorization": f"Bearer {wrong_key}"})
        logs = "\n".join(captured.output)
        stored = self.db.dump()
        for secret in (self.key, wrong_key, hash_api_key(self.key)):
            self.assertNotIn(secret, logs)
            self.assertNotIn(secret, stored)

    def test_rejected_requests_logged_as_unauthorized(self):
        self.assert_rejected({})
        self.assert_rejected({API_KEY_HEADER: generate_api_key()})
        stored = json.loads(self.db.dump())
        self.assertEqual(len(stored), 2)
        self.assertTrue(all("unauthorized" in row and "401" in row for row in stored))


if __name__ == "__main__":
    unittest.main()
