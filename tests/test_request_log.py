"""Tests for the SQL request logger (REQ-14, NFR-7).

The unit tests run the SQLAlchemy Core code against isolated in-memory
SQLite engines. That checks the insert/purge logic and the schema, but it is
NOT a PostgreSQL test.

PostgreSQL integration tests run only when TRUTHLAYER_TEST_DATABASE_URL is
set, and are fail-closed (see IsolatedPostgres):
- the URL must name a dedicated database ending in `_test`, differ from
  TRUTHLAYER_DATABASE_URL, and carry no connection-redirecting options;
  this is checked before any engine is created;
- after connecting, current_database() must match that name;
- every test runs inside a freshly created, randomly named schema (pinned
  via search_path and re-checked with current_schema()), which is dropped
  afterwards. The production purge therefore only ever sees test rows.
"""
import os
import re
import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, delete, func, inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateSchema, DropSchema

from api.request_log import RequestLogEntry, SqlRequestLogger, request_logs

TEST_PATH = "/__truthlayer_test__"

# --------------------------------------------------------------------------- #
# PostgreSQL test-database safety guard
# --------------------------------------------------------------------------- #

SAFE_TEST_DB_NAME_RE = re.compile(r"[a-z0-9_]+_test")
# Query options that cannot redirect the connection to another database or schema.
ALLOWED_QUERY_KEYS = frozenset({"sslmode", "connect_timeout", "application_name"})


class UnsafeTestDatabaseError(RuntimeError):
    """The configured test database is not clearly a dedicated test database."""


def _db_identity(url: URL) -> tuple[str, int, str | None]:
    return (url.host or "localhost", url.port or 5432, url.database)


def validate_test_database_url(url: str, app_database_url: str | None = None) -> URL:
    """Parse (without connecting) and reject anything not clearly a test DB.

    Error messages never include the URL, so credentials cannot leak.
    """
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise UnsafeTestDatabaseError("TRUTHLAYER_TEST_DATABASE_URL is not a valid database URL") from None

    if parsed.get_backend_name() != "postgresql":
        raise UnsafeTestDatabaseError("TRUTHLAYER_TEST_DATABASE_URL must be a postgresql:// URL")

    name = parsed.database or ""
    if not SAFE_TEST_DB_NAME_RE.fullmatch(name):
        raise UnsafeTestDatabaseError(
            f"Refusing database {name!r}: test database names must match "
            f"{SAFE_TEST_DB_NAME_RE.pattern!r} (e.g. 'truthlayer_test')"
        )

    extra = sorted(set(parsed.query) - ALLOWED_QUERY_KEYS)
    if extra:
        raise UnsafeTestDatabaseError(f"Refusing URL query options {extra}; allowed: {sorted(ALLOWED_QUERY_KEYS)}")

    if app_database_url:
        try:
            app_url = make_url(app_database_url)
        except ArgumentError:
            app_url = None
        if app_url is not None and _db_identity(app_url) == _db_identity(parsed):
            raise UnsafeTestDatabaseError("Test database is the same as TRUTHLAYER_DATABASE_URL")

    return parsed


def assert_connected_database(conn, expected: str) -> None:
    actual = conn.execute(text("SELECT current_database()")).scalar_one()
    if actual != expected:
        raise UnsafeTestDatabaseError(f"Connected to {actual!r}, expected {expected!r}")


def assert_current_schema(conn, expected: str) -> None:
    actual = conn.execute(text("SELECT current_schema()")).scalar_one_or_none()
    if actual != expected:
        raise UnsafeTestDatabaseError(f"current_schema() is {actual!r}, expected {expected!r}")


class IsolatedPostgres:
    """Throwaway schemas inside a validated test database.

    Validation happens in __init__, before any engine is created; start()
    verifies the live connection before creating anything.
    """

    def __init__(self, url: str, app_database_url: str | None = None) -> None:
        self.url = validate_test_database_url(url, app_database_url)
        self.schema = f"tl_test_{uuid.uuid4().hex[:12]}"
        self.sentinel_schema = f"{self.schema}_sentinel"
        self._admin = None
        self._engines = []

    def start(self) -> None:
        self._admin = create_engine(self.url)
        with self._admin.begin() as conn:
            assert_connected_database(conn, self.url.database)
            conn.execute(CreateSchema(self.schema))
            conn.execute(CreateSchema(self.sentinel_schema))

    def engine_for(self, schema: str):
        engine = create_engine(self.url, connect_args={"options": f"-c search_path={schema}"})
        self._engines.append(engine)
        with engine.connect() as conn:
            assert_connected_database(conn, self.url.database)
            assert_current_schema(conn, schema)
        return engine

    def close(self) -> None:
        for engine in self._engines:
            engine.dispose()
        if self._admin is not None:
            with self._admin.begin() as conn:
                conn.execute(DropSchema(self.schema, cascade=True, if_exists=True))
                conn.execute(DropSchema(self.sentinel_schema, cascade=True, if_exists=True))
            self._admin.dispose()


# --------------------------------------------------------------------------- #
# Logger contract (runs on SQLite always, on PostgreSQL when configured)
# --------------------------------------------------------------------------- #

def _entry(**overrides):
    values = dict(
        method="POST", path=TEST_PATH, status_code=200, processing_time_ms=12,
        input_word_count=10, input_char_count=60, knowledge_base_id="wikipedia_subset",
        claim_count=2, supported_count=1, contradicted_count=0, unverifiable_count=1,
    )
    values.update(overrides)
    return RequestLogEntry(**values)


def _days_ago(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


def _count(engine):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(request_logs)).scalar_one()


class SqlLoggerContract:
    """Shared checks. Subclasses set self.engine (isolated store under test)
    and self.sentinel_engine (a separate store holding unrelated rows)."""

    def setUp(self):
        self.logger = SqlRequestLogger(self.engine)
        self.logger.create_schema()
        self.sentinel = SqlRequestLogger(self.sentinel_engine)
        self.sentinel.create_schema()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        # Safe: both engines point at isolated, test-only stores.
        for engine in (self.engine, self.sentinel_engine):
            with engine.begin() as conn:
                conn.execute(delete(request_logs))

    def _rows(self):
        with self.engine.connect() as conn:
            return conn.execute(select(request_logs).order_by(request_logs.c.id)).mappings().all()

    def test_record_inserts_metadata_row(self):
        self.logger.record(_entry())
        self.logger.record(_entry(status_code=401, error_type="unauthorized", claim_count=None))
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status_code"], 200)
        self.assertEqual(rows[0]["supported_count"], 1)
        self.assertEqual(rows[1]["error_type"], "unauthorized")
        self.assertIsNone(rows[1]["claim_count"])

    def test_purge_removes_expired_and_keeps_fresh(self):
        self.logger.record(_entry(created_at=_days_ago(40), status_code=500))
        self.logger.record(_entry(created_at=_days_ago(31), status_code=401))
        self.logger.record(_entry(created_at=_days_ago(29), status_code=400))
        self.logger.record(_entry(status_code=200))
        self.assertEqual(self.logger.purge_older_than(30), 2)
        self.assertEqual(sorted(r["status_code"] for r in self._rows()), [200, 400])

    def test_purge_cannot_touch_unrelated_records(self):
        self.sentinel.record(_entry(created_at=_days_ago(365), path="/unrelated"))
        self.sentinel.record(_entry(created_at=_days_ago(90), path="/unrelated"))
        self.logger.record(_entry(created_at=_days_ago(40)))
        self.assertEqual(self.logger.purge_older_than(30), 1)
        self.assertEqual(_count(self.engine), 0)
        self.assertEqual(_count(self.sentinel_engine), 2)

    def test_schema_has_no_content_columns(self):
        columns = {c["name"] for c in inspect(self.engine).get_columns("request_logs")}
        for forbidden in ("text", "input_text", "claims", "evidence", "api_key", "client_ip"):
            self.assertNotIn(forbidden, columns)

    def test_create_schema_is_idempotent(self):
        self.logger.create_schema()


class TestSqlRequestLoggerOnSqlite(SqlLoggerContract, unittest.TestCase):
    def setUp(self):
        # Two independent in-memory databases: isolated by construction.
        self.engine = create_engine("sqlite://", poolclass=StaticPool)
        self.sentinel_engine = create_engine("sqlite://", poolclass=StaticPool)
        super().setUp()


@unittest.skipUnless(
    os.environ.get("TRUTHLAYER_TEST_DATABASE_URL"),
    "set TRUTHLAYER_TEST_DATABASE_URL to a dedicated *_test PostgreSQL database to run",
)
class TestSqlRequestLoggerOnPostgres(SqlLoggerContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Raises UnsafeTestDatabaseError (test errors, not skips) if unsafe.
        cls.pg = IsolatedPostgres(
            os.environ["TRUTHLAYER_TEST_DATABASE_URL"],
            os.environ.get("TRUTHLAYER_DATABASE_URL"),
        )
        cls.addClassCleanup(cls.pg.close)
        cls.pg.start()
        cls.pg_engine = cls.pg.engine_for(cls.pg.schema)
        cls.pg_sentinel_engine = cls.pg.engine_for(cls.pg.sentinel_schema)

    def setUp(self):
        self.engine = self.pg_engine
        self.sentinel_engine = self.pg_sentinel_engine
        super().setUp()


# --------------------------------------------------------------------------- #
# Guard tests (no database connection; engine creation is mocked out)
# --------------------------------------------------------------------------- #

SAFE_URL = "postgresql+psycopg://tl:s3cr3t-pw@localhost:5432/truthlayer_test"


class TestTestDatabaseGuard(unittest.TestCase):
    def test_accepts_dedicated_test_database(self):
        self.assertEqual(validate_test_database_url(SAFE_URL).database, "truthlayer_test")
        validate_test_database_url(SAFE_URL + "?sslmode=disable&connect_timeout=3")

    def test_rejects_non_test_database_names(self):
        for name in ("truthlayer", "truthlayer_dev", "postgres", "template1", "test",
                     "mytest", "truthlayer_test_backup", "TruthLayer_test", "truthlayer-test", ""):
            with self.subTest(name=name), self.assertRaises(UnsafeTestDatabaseError):
                validate_test_database_url(f"postgresql+psycopg://tl:pw@localhost/{name}")

    def test_rejects_non_postgres_and_invalid_urls(self):
        for url in ("sqlite:///truthlayer_test", "mysql://u:p@h/truthlayer_test", "not a url"):
            with self.subTest(url=url), self.assertRaises(UnsafeTestDatabaseError):
                validate_test_database_url(url)

    def test_rejects_redirecting_query_options(self):
        for query in ("options=-csearch_path%3Dpublic", "service=prod", "dbname=truthlayer", "host=prod-db"):
            with self.subTest(query=query), self.assertRaises(UnsafeTestDatabaseError):
                validate_test_database_url(f"{SAFE_URL}?{query}")

    def test_rejects_same_database_as_application(self):
        with self.assertRaises(UnsafeTestDatabaseError):
            validate_test_database_url(SAFE_URL, app_database_url="postgresql://other:pw@localhost/truthlayer_test")
        # Same name on a different server is a different database.
        validate_test_database_url(SAFE_URL, app_database_url="postgresql://u:p@db.example:5432/truthlayer_test")

    def test_error_messages_do_not_leak_credentials(self):
        for url in ("postgresql+psycopg://tl:s3cr3t-pw@localhost/truthlayer",
                    "postgresql+psycopg://tl:s3cr3t-pw@localhost/truthlayer_test?service=x",
                    "s3cr3t-pw not a url"):
            with self.subTest(url=url), self.assertRaises(UnsafeTestDatabaseError) as ctx:
                validate_test_database_url(url)
            self.assertNotIn("s3cr3t-pw", str(ctx.exception))

    def test_unsafe_config_rejected_before_any_engine_or_ddl(self):
        with mock.patch.object(sys.modules[__name__], "create_engine") as fake_create_engine:
            with self.assertRaises(UnsafeTestDatabaseError):
                IsolatedPostgres("postgresql+psycopg://tl:pw@localhost/truthlayer")
        fake_create_engine.assert_not_called()

    def test_wrong_connected_database_rejected_before_schema_creation(self):
        conn = mock.MagicMock()
        conn.execute.return_value.scalar_one.return_value = "truthlayer"  # e.g. a pooler alias
        engine = mock.MagicMock()
        engine.begin.return_value.__enter__.return_value = conn
        with mock.patch.object(sys.modules[__name__], "create_engine", return_value=engine):
            pg = IsolatedPostgres(SAFE_URL)
            with self.assertRaises(UnsafeTestDatabaseError):
                pg.start()
        self.assertEqual(conn.execute.call_count, 1)  # only SELECT current_database(); no CREATE SCHEMA

    def test_wrong_current_schema_rejected(self):
        conn = mock.MagicMock()
        conn.execute.return_value.scalar_one_or_none.return_value = "public"
        with self.assertRaises(UnsafeTestDatabaseError):
            assert_current_schema(conn, "tl_test_abc")

    def test_isolated_schema_names_are_unique_and_test_prefixed(self):
        a, b = IsolatedPostgres(SAFE_URL), IsolatedPostgres(SAFE_URL)
        self.assertNotEqual(a.schema, b.schema)
        self.assertRegex(a.schema, r"^tl_test_[0-9a-f]{12}$")
        self.assertEqual(a.sentinel_schema, a.schema + "_sentinel")


if __name__ == "__main__":
    unittest.main()
