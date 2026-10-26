"""Request usage logging to PostgreSQL (REQ-14, NFR-7, SRS section 7).

Privacy: only metadata is stored — never the submitted text, claims, evidence,
API key, or client IP. Rows older than the configured retention period are
purged on startup.

The logger is injected into the app, so tests can use an in-memory fake and
never need production database credentials.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    create_engine,
    delete,
    insert,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

metadata = MetaData()

request_logs = Table(
    "request_logs",
    metadata,
    # BIGSERIAL on PostgreSQL; INTEGER so SQLite autoincrements in tests.
    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("method", String(8), nullable=False),
    Column("path", String(128), nullable=False),
    Column("status_code", SmallInteger, nullable=False),
    Column("processing_time_ms", Integer, nullable=False),
    Column("input_word_count", Integer),
    Column("input_char_count", Integer),
    Column("knowledge_base_id", String(64)),
    Column("claim_count", Integer),
    Column("supported_count", Integer),
    Column("contradicted_count", Integer),
    Column("unverifiable_count", Integer),
    Column("error_type", String(64)),
)


@dataclass
class RequestLogEntry:
    method: str
    path: str
    status_code: int
    processing_time_ms: int
    input_word_count: Optional[int] = None
    input_char_count: Optional[int] = None
    knowledge_base_id: Optional[str] = None
    claim_count: Optional[int] = None
    supported_count: Optional[int] = None
    contradicted_count: Optional[int] = None
    unverifiable_count: Optional[int] = None
    error_type: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RequestLogger(Protocol):
    def record(self, entry: RequestLogEntry) -> None: ...

    def purge_older_than(self, days: int) -> int: ...


class SqlRequestLogger:
    """Writes RequestLogEntry rows via SQLAlchemy Core.

    Production uses PostgreSQL (postgresql+psycopg://...). The table uses only
    portable column types, so the same code can be exercised against an
    isolated SQLite engine in unit tests.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, database_url: str) -> "SqlRequestLogger":
        return cls(create_engine(database_url, pool_pre_ping=True))

    def create_schema(self) -> None:
        metadata.create_all(self._engine)

    def record(self, entry: RequestLogEntry) -> None:
        with self._engine.begin() as conn:
            conn.execute(insert(request_logs).values(**asdict(entry)))

    def purge_older_than(self, days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._engine.begin() as conn:
            result = conn.execute(delete(request_logs).where(request_logs.c.created_at < cutoff))
        return result.rowcount or 0

    def dispose(self) -> None:
        self._engine.dispose()
