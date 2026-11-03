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