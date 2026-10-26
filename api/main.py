"""TruthLayer FastAPI application: POST /verify (REQ-12 to REQ-15).

Run (from the repo root):
    uvicorn api.main:create_app --factory --env-file .env --port 8000

`create_app` accepts injected settings, pipeline and request logger so tests
run with fake models and no database credentials.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from claim_extraction import InputTooLongError, count_words
from evidence_verification import EvidenceVerifier, load_knowledge_base

from api.auth import API_KEY_HEADER, is_valid_api_key
from api.config import Settings
from api.pipeline import VerificationPipeline, verdict_summary
from api.request_log import RequestLogEntry, RequestLogger, SqlRequestLogger
from api.schemas import ErrorResponse, VerifyRequest, VerifyResponse

logger = logging.getLogger("truthlayer.api")

LOGGED_PATHS = frozenset({"/verify"})

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Malformed input, input too long, or unknown knowledge_base_id."},
    401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key header."},
    500: {"model": ErrorResponse, "description": "Internal verification pipeline failure."},
}


def _log_fields(request: Request) -> dict[str, Any]:
    """Per-request metadata the handler fills in for the request log."""
    if not hasattr(request.state, "log_fields"):
        request.state.log_fields = {}
    return request.state.log_fields


def require_api_key(request: Request, api_key: str | None = Depends(_api_key_header)) -> None:
    settings: Settings = request.app.state.settings
    if not is_valid_api_key(api_key, settings.api_key_hashes):
        _log_fields(request)["error_type"] = "unauthorized"
        raise HTTPException(status_code=401, detail=f"Provide a valid key in the {API_KEY_HEADER} header.")


def _format_validation_error(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        field = ".".join(loc)
        parts.append(f"{field}: {err.get('msg')}" if field else str(err.get("msg")))
    return "Invalid request: " + "; ".join(parts)


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: VerificationPipeline | None = None,
    request_logger: RequestLogger | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            kb = load_knowledge_base(settings.knowledge_base_path)
            # Loads (and on first run downloads) the embedding and NLI models.
            app.state.pipeline = VerificationPipeline(EvidenceVerifier(kb))
        if app.state.request_logger is None:
            sql_logger = SqlRequestLogger.from_url(settings.database_url)
            sql_logger.create_schema()
            app.state.request_logger = sql_logger
        purged = app.state.request_logger.purge_older_than(settings.log_retention_days)
        if purged:
            logger.info("Purged %d request log rows older than %d days", purged, settings.log_retention_days)
        yield
        dispose = getattr(app.state.request_logger, "dispose", None)
        if dispose:
            dispose()

    app = FastAPI(
        title="TruthLayer API",
        version="0.1.0",
        description="Claim-level verification of AI-generated text (Supported / Contradicted / Unverifiable).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.request_logger = request_logger

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # REQ-15: malformed payloads are 400 (FastAPI's default would be 422).
        _log_fields(request).setdefault("error_type", "invalid_request")
        return JSONResponse(status_code=400, content={"detail": _format_validation_error(exc)})

    default_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        # Validation errors are 400 (above), so drop FastAPI's auto-documented 422.
        schema = default_openapi()
        for operations in schema.get("paths", {}).values():
            for operation in operations.values():
                operation.get("responses", {}).pop("422", None)
        return schema

    app.openapi = openapi

    @app.post(
        "/verify",
        response_model=VerifyResponse,
        responses=_ERROR_RESPONSES,
        dependencies=[Depends(require_api_key)],
    )
    def verify(body: VerifyRequest, request: Request) -> VerifyResponse:
        fields = _log_fields(request)
        fields["input_char_count"] = len(body.text)
        fields["input_word_count"] = count_words(body.text)

        kb_id = body.knowledge_base_id or settings.knowledge_base_id
        if kb_id != settings.knowledge_base_id:
            fields["error_type"] = "unknown_knowledge_base"
            raise HTTPException(
                status_code=400,
                detail=f"Unknown knowledge_base_id. Available: {settings.knowledge_base_id}.",
            )
        fields["knowledge_base_id"] = kb_id

        started = time.perf_counter()
        try:
            status, message, claims = request.app.state.pipeline.run(body.text)
        except InputTooLongError as exc:
            fields["error_type"] = "input_too_long"
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception:
            fields["error_type"] = "pipeline_error"
            logger.exception("Verification pipeline failed")
            raise HTTPException(status_code=500, detail="The verification pipeline failed.") from None
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        summary = verdict_summary(claims)
        fields.update(
            claim_count=len(claims),
            supported_count=summary["supported"],
            contradicted_count=summary["contradicted"],
            unverifiable_count=summary["unverifiable"],
        )
        return VerifyResponse(status=status, message=message, claims=claims, processing_time_ms=elapsed_ms)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # REQ-14: one metadata row per /verify request, including rejected ones.
        if request.url.path not in LOGGED_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        fields = _log_fields(request)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            if status_code >= 400:
                fields.setdefault("error_type", "invalid_request" if status_code < 500 else "server_error")
            entry = RequestLogEntry(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                processing_time_ms=round((time.perf_counter() - started) * 1000),
                **fields,
            )
            try:
                await run_in_threadpool(request.app.state.request_logger.record, entry)
            except Exception as exc:
                # Never fail the client's request because logging failed;
                # log only the exception type, not the DB URL.
                logger.error("Failed to write request log entry: %s", type(exc).__name__)

    # Added last so it is outermost: error responses also carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["POST"],
        allow_headers=["Content-Type", API_KEY_HEADER],
    )
    return app
