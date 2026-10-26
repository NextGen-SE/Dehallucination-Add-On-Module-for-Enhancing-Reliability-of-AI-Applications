"""Environment-driven settings for the TruthLayer API.

All configuration comes from environment variables (see `.env.example`).
Secrets are never given defaults: the API refuses to start without API key
hashes or a database URL, rather than silently running unauthenticated or
without the request log required by REQ-14.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KNOWLEDGE_BASE_ID = "wikipedia_subset"  # SRS Appendix B default
DEFAULT_KNOWLEDGE_BASE_PATH = REPO_ROOT / "knowledge_base.json"
DEFAULT_CORS_ORIGINS = ("http://localhost:3000",)
DEFAULT_LOG_RETENTION_DAYS = 30

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    api_key_hashes: tuple[str, ...]
    database_url: str
    knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID
    knowledge_base_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    log_retention_days: int = DEFAULT_LOG_RETENTION_DAYS

    # Custom repr so the database password never lands in logs or tracebacks.
    def __repr__(self) -> str:
        return (
            f"Settings(api_key_hashes=<{len(self.api_key_hashes)} configured>, "
            f"database_url=<redacted>, knowledge_base_id={self.knowledge_base_id!r}, "
            f"knowledge_base_path={str(self.knowledge_base_path)!r}, "
            f"cors_origins={self.cors_origins!r}, "
            f"log_retention_days={self.log_retention_days})"
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        hashes = tuple(h.lower() for h in _split_csv(env.get("TRUTHLAYER_API_KEY_HASHES", "")))
        if not hashes:
            raise ConfigError(
                "TRUTHLAYER_API_KEY_HASHES is not set. Generate a key with "
                "`python -m api.auth` and set its hash."
            )
        bad = [h for h in hashes if not _SHA256_HEX_RE.match(h)]
        if bad:
            raise ConfigError(
                "TRUTHLAYER_API_KEY_HASHES must contain SHA-256 hex digests "
                f"(64 hex chars); {len(bad)} invalid entr{'y' if len(bad) == 1 else 'ies'}."
            )

        database_url = env.get("TRUTHLAYER_DATABASE_URL", "").strip()
        if not database_url:
            raise ConfigError(
                "TRUTHLAYER_DATABASE_URL is not set. A PostgreSQL URL such as "
                "postgresql+psycopg://user:password@localhost:5432/truthlayer "
                "is required for request logging (REQ-14)."
            )

        kb_path = Path(env.get("TRUTHLAYER_KNOWLEDGE_BASE_PATH", "") or DEFAULT_KNOWLEDGE_BASE_PATH)
        if not kb_path.is_absolute():
            kb_path = REPO_ROOT / kb_path

        retention_raw = env.get("TRUTHLAYER_LOG_RETENTION_DAYS", "") or str(DEFAULT_LOG_RETENTION_DAYS)
        try:
            retention = int(retention_raw)
        except ValueError:
            raise ConfigError("TRUTHLAYER_LOG_RETENTION_DAYS must be an integer") from None
        if retention < 1:
            raise ConfigError("TRUTHLAYER_LOG_RETENTION_DAYS must be at least 1")

        return cls(
            api_key_hashes=hashes,
            database_url=database_url,
            knowledge_base_id=env.get("TRUTHLAYER_KNOWLEDGE_BASE_ID", "").strip()
            or DEFAULT_KNOWLEDGE_BASE_ID,
            knowledge_base_path=kb_path,
            cors_origins=_split_csv(env.get("TRUTHLAYER_CORS_ORIGINS", ""))
            or DEFAULT_CORS_ORIGINS,
            log_retention_days=retention,
        )
