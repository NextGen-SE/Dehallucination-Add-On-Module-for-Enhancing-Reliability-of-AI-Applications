"""End-to-end request logging tests for Issue #17 (SRS REQ-14, NFR-7).

Drives POST /verify through the real middleware into the real SQL request
logger (in-memory SQLite), then reads rows back with
SqlRequestLogger.recent(). Complements tests/test_api.py (fake logger) and
tests/test_request_log.py (logger SQL in isolation). Fake models; no
downloads or database credentials needed.

Run:  python -m pytest tests/test_request_logging.py -v
"""
import logging
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from api.auth import API_KEY_HEADER, generate_api_key, hash_api_key
from api.config import Settings
from api.main import create_app
from api.pipeline import VerificationPipeline
from api.request_log import RequestLogEntry, SqlRequestLogger
from evidence_verification import EvidenceVerifier

SENTINEL = "ZEBRA-SECRET-17"
KB = [
    {"source_id": "KB-001", "text": "The Earth orbits the Sun once approximately every 365.25 days."},
    {"source_id": "KB-002", "text": "Water is made of hydrogen and oxygen."},
]
# One Supported, one Contradicted, one Unverifiable claim (see fake_nli).
MIXED = "The Earth orbits the Sun. The Sun orbits the Earth. Bananas are a good source of potassium."


def fake_embedder(texts):
    out = []
    for t in texts:
        low = t.lower()
        astro, chem = ("earth" in low or "sun" in low), ("water" in low or "hydrogen" in low)
        out.append([float(astro), float(chem), 0.0 if (astro or chem) else 1.0])
    return out


def fake_nli(premise, hypothesis):
    if hypothesis.startswith("The Earth orbits") or hypothesis.startswith("Water"):
        return {"entailment": 0.95, "contradiction": 0.03, "neutral": 0.02}
    if hypothesis.startswith("The Sun orbits"):
        return {"entailment": 0.04, "contradiction": 0.9, "neutral": 0.06}
    return {"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8}


def make_pipeline():
    return VerificationPipeline(EvidenceVerifier(KB, embedder=fake_embedder, nli_classifier=fake_nli))


class SlowPipeline:
    """Real pipeline that sleeps first and records when it started running."""

    DELAY_S = 0.15

    def __init__(self):
        self.inner = make_pipeline()
        self.started_at = None

    def run(self, text):
        self.started_at = datetime.now(timezone.utc)
        time.sleep(self.DELAY_S)
        return self.inner.run(text)


class ExplodingPipeline:
    def run(self, text):
        raise RuntimeError("model crashed")


def sqlite_logger():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    log = SqlRequestLogger(engine)
    log.create_schema()
    return log


class LoggingTestCase(unittest.TestCase):
    retention_days = 30

    def setUp(self):
        self.key = generate_api_key()
        self.db = sqlite_logger()

    def client(self, pipeline=None, db=None):
        settings = Settings(
            api_key_hashes=(hash_api_key(self.key),),
            database_url="postgresql+psycopg://user:dbpassword@unused/db",
            log_retention_days=self.retention_days,
        )
        app = create_app(settings, pipeline=pipeline or make_pipeline(), request_logger=db or self.db)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def verify(self, client, text_value, **extra):
        return client.post("/verify", json={"text": text_value, **extra}, headers={API_KEY_HEADER: self.key})

    def only_entry(self) -> RequestLogEntry:
        entries = self.db.recent()
        self.assertEqual(len(entries), 1)
        return entries[0]


class TestRecordedMetadata(LoggingTestCase):
    def test_success_row_round_trips_with_summary_and_lengths(self):
        resp = self.verify(self.client(), MIXED)
        self.assertEqual(resp.status_code, 200)
        entry = self.only_entry()
        self.assertEqual((entry.method, entry.path, entry.status_code), ("POST", "/verify", 200))
        self.assertEqual(entry.input_char_count, len(MIXED))
        self.assertEqual(entry.input_word_count, 17)
        self.assertEqual(entry.knowledge_base_id, "wikipedia_subset")
        self.assertEqual(
            (entry.claim_count, entry.supported_count, entry.contradicted_count, entry.unverifiable_count),
            (3, 1, 1, 1),
        )
        self.assertIsNone(entry.error_type)

    def test_summary_counts_match_response_verdicts(self):
        resp = self.verify(self.client(), MIXED)
        verdicts = [c["verdict"] for c in resp.json()["claims"]]
        entry = self.only_entry()
        self.assertEqual(entry.claim_count, len(verdicts))
        self.assertEqual(entry.supported_count, verdicts.count("Supported"))
        self.assertEqual(entry.contradicted_count, verdicts.count("Contradicted"))
        self.assertEqual(entry.unverifiable_count, verdicts.count("Unverifiable"))

    def test_all_supported_outcome(self):
        self.verify(self.client(), "The Earth orbits the Sun. Water is made of hydrogen and oxygen.")
        entry = self.only_entry()
        self.assertEqual(
            (entry.claim_count, entry.supported_count, entry.contradicted_count, entry.unverifiable_count),
            (2, 2, 0, 0),
        )

    def test_no_verifiable_claims_records_zero_counts(self):
        resp = self.verify(self.client(), "I think this is wonderful.")
        self.assertEqual(resp.json()["status"], "no_verifiable_claims")
        entry = self.only_entry()
        self.assertEqual(entry.status_code, 200)
        self.assertEqual(
            (entry.claim_count, entry.supported_count, entry.contradicted_count, entry.unverifiable_count),
            (0, 0, 0, 0),
        )

    def test_pipeline_failure_records_error_without_summary(self):
        resp = self.verify(self.client(pipeline=ExplodingPipeline()), MIXED)
        self.assertEqual(resp.status_code, 500)
        entry = self.only_entry()
        self.assertEqual((entry.status_code, entry.error_type), (500, "pipeline_error"))
        self.assertEqual(entry.input_char_count, len(MIXED))
        self.assertIsNone(entry.claim_count)
        self.assertIsNone(entry.supported_count)

    def test_unauthorized_request_records_no_input_metadata(self):
        resp = self.client().post("/verify", json={"text": MIXED})
        self.assertEqual(resp.status_code, 401)
        entry = self.only_entry()
        self.assertEqual((entry.status_code, entry.error_type), (401, "unauthorized"))
        self.assertIsNone(entry.input_char_count)


class TestTiming(LoggingTestCase):
    def test_timestamp_is_request_arrival_time_in_utc(self):
        pipeline = SlowPipeline()
        before = datetime.now(timezone.utc)
        self.verify(self.client(pipeline=pipeline), MIXED)
        after = datetime.now(timezone.utc)
        entry = self.only_entry()
        self.assertEqual(entry.created_at.utcoffset(), timedelta(0))
        self.assertGreaterEqual(entry.created_at, before)
        # Stamped when the request arrived, not after the slow pipeline finished.
        self.assertLessEqual(entry.created_at, pipeline.started_at)
        self.assertLessEqual(entry.created_at, after)

    def test_processing_time_reflects_request_duration(self):
        pipeline = SlowPipeline()
        started = time.perf_counter()
        resp = self.verify(self.client(pipeline=pipeline), MIXED)
        wall_ms = (time.perf_counter() - started) * 1000
        entry = self.only_entry()
        self.assertGreaterEqual(entry.processing_time_ms, int(SlowPipeline.DELAY_S * 1000))
        self.assertLessEqual(entry.processing_time_ms, wall_ms + 1)
        # The middleware's total includes the pipeline time reported to the client.
        self.assertGreaterEqual(entry.processing_time_ms, resp.json()["processing_time_ms"])


class TestRetrieval(LoggingTestCase):
    def test_recent_returns_newest_first_and_respects_limit(self):
        client = self.client()
        self.verify(client, "The Earth orbits the Sun.")
        self.verify(client, "I think this is wonderful.")
        client.post("/verify", json={"text": MIXED})  # 401
        entries = self.db.recent()
        self.assertEqual([e.status_code for e in entries], [401, 200, 200])
        self.assertEqual([e.claim_count for e in entries], [None, 0, 1])
        self.assertEqual([e.status_code for e in self.db.recent(limit=1)], [401])

    def test_recent_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError):
            self.db.recent(limit=0)


class TestRetention(LoggingTestCase):
    retention_days = 7

    def test_startup_purges_rows_older_than_configured_retention(self):
        now = datetime.now(timezone.utc)
        for age_days, status in ((30, 500), (8, 401), (6, 400), (0, 200)):
            self.db.record(RequestLogEntry(
                method="POST", path="/verify", status_code=status, processing_time_ms=1,
                created_at=now - timedelta(days=age_days),
            ))
        self.client()  # lifespan startup runs the purge with retention_days=7
        self.assertEqual(sorted(e.status_code for e in self.db.recent()), [200, 400])


class TestDatabaseFailures(LoggingTestCase):
    def test_sql_failure_keeps_request_successful_and_leaks_nothing(self):
        broken = sqlite_logger()
        client = self.client(db=broken)  # starts up (and purges) normally
        with broken._engine.begin() as conn:
            conn.execute(text("DROP TABLE request_logs"))  # DB breaks at runtime: every INSERT raises
        text_value = f"The Earth orbits the Sun. {SENTINEL} is a project code name."
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("truthlayer.test").debug("capture-start")
            resp = self.verify(client, text_value)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        logs = "\n".join(captured.output)
        self.assertIn("Failed to write request log entry: OperationalError", logs)
        # SQLAlchemy errors embed the SQL statement and parameters; none may surface.
        for leaked in (SENTINEL, self.key, "dbpassword", "INSERT", "request_logs"):
            self.assertNotIn(leaked, logs)


if __name__ == "__main__":
    unittest.main()
