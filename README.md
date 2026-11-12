# TruthLayer — Claim Extraction (Issues #1–#3) (Person-1)

This folder implements the initial scope of:
- Issue #1: Atomic candidate claim extraction
- Issue #2: Claim-to-source-sentence mapping
- Issue #3: Unit tests

## Important limitation
This is a lightweight, dependency-free baseline using Python's standard library. It uses sentence splitting and conservative heuristics to identify candidate factual statements. It is not a full NLP claim detector, cannot guarantee true factuality or perfect atomicity, and may miss claims or misclassify some statements. It is intended as a working starting point that the team can improve and integrate.

## Files
- `claim_extraction.py`: extraction interface and structured results
- `tests/test_claim_extraction.py`: unit tests
- `examples/demo.py`: runnable example
- `requirements.txt`: optional pytest dependency
- `pyproject.toml`: pytest configuration

## Run the demo
```bash
python examples/demo.py
```

## Run tests without installing packages
```bash
python -m unittest discover -s tests -v
```

Optional pytest:
```bash
python -m pip install pytest
python -m pytest
```

## Example API
```python
from claim_extraction import extract_claims

result = extract_claims("The Earth orbits the Sun. Water contains hydrogen and oxygen.")
print(result.to_dict())
```

Each claim includes `claim_id`, `text`, `source_sentence`, `start_char`, `end_char`, `source_sentence_start`, and `source_sentence_end`.

## Requirement mapping
- REQ-1: accepts up to 2,000 words; raises `InputTooLongError` above the limit.
- REQ-2: extracts candidate factual statements and filters some obvious opinion/rhetorical forms with heuristics.
- REQ-3: preserves the source sentence and character offsets.
- REQ-4: returns `status="no_verifiable_claims"` and the message `No verifiable claims found` if none are identified.

Confirm exact status/error conventions with the API owner before integration.

## Evidence Retrieval and NLI Verification (Person 2)

Takes the `Claim` objects produced by `claim_extraction.py` and decides,
for each one, whether it is **Supported**, **Contradicted**, or
**Unverifiable** against a small knowledge base.

### How it works
1. **Retrieve** (REQ-5): embed the claim and every knowledge-base passage
   with `sentence-transformers/all-MiniLM-L6-v2`, rank passages by cosine
   similarity, keep the top-k (default 5).
2. **Threshold check** (REQ-8): if even the best-ranked passage falls below
   `relevance_threshold` (default 0.35), the claim is marked
   **Unverifiable** immediately — the NLI model is never called, since
   there's nothing relevant to check it against.
3. **Classify** (REQ-6, REQ-7): otherwise, the single best passage and the
   claim are passed to an NLI model
   (`cross-encoder/nli-deberta-v3-small`, via a HuggingFace
   `text-classification` pipeline) which scores `entailment` /
   `contradiction` / `neutral`. The highest-scoring label becomes the
   verdict, and its score becomes the confidence (0.0–1.0).
4. **Evidence** (REQ-10): the supporting/contradicting passage's exact
   text and `source_id` are attached to the result. A `neutral` outcome
   returns **Unverifiable** with no evidence — the model found something
   topically relevant but inconclusive.

### Important limitation
This is a **single-best-passage** baseline: it checks the claim against
only the top-ranked passage, not all `top_k` retrieved ones. That's a
known simplification for this sprint — multi-passage aggregation (e.g.
taking the best entailment/contradiction score across all retrieved
passages, not just the first) is a natural follow-up once we start tuning
against the REQ-19 accuracy target. The relevance threshold (0.35) is also
a starting guess, not an empirically tuned value yet.

### Files
- `evidence_verification.py`: `EvidenceVerifier`, `EvidencePassage`,
  `VerificationResult`, `load_knowledge_base`
- `knowledge_base.json`: small starter knowledge base (swap via REQ-9 by
  pointing `load_knowledge_base` at a different file with the same shape)
- `tests/test_evidence_verification.py`: unit tests against fake
  embedder/classifier (fast, no network), plus one real-model integration
  test that auto-skips if HuggingFace Hub isn't reachable
- `examples/demo_verification.py`: runs Person 1's extractor into Person
  2's verifier end-to-end

### Run the demo
```bash
pip install -r requirements.txt
python examples/demo_verification.py
```

### Run tests
```bash
python -m unittest discover -s tests -v
```
The real-model integration test only runs if `sentence-transformers` /
`transformers` are installed **and** HuggingFace Hub is reachable; it's
skipped automatically otherwise so the rest of the suite never blocks on
network access.

### Requirement mapping
- REQ-5: top-k semantic retrieval over the knowledge base.
- REQ-6: Supported / Contradicted / Unverifiable classification via NLI.
- REQ-7: numeric confidence (0.0–1.0) attached to every verdict.
- REQ-8: Unverifiable when no retrieved passage clears the relevance threshold.
- REQ-9: knowledge base is swappable — just point at a different JSON file.
- REQ-10: exact evidence passage text + source_id returned for Supported/Contradicted.

## Verification API (Person 4)

FastAPI service exposing the extraction → verification pipeline as
`POST /verify`, with API-key auth and PostgreSQL request logging.

### Files
- `api/main.py`: `create_app()` factory, `/verify` route, error handlers, CORS, request-log middleware
- `api/schemas.py`: request/response models (SRS Appendix B)
- `api/pipeline.py`: calls `extract_claims` then `EvidenceVerifier.verify_claims`, merges by `claim_id`
- `api/auth.py`: hashed `X-API-Key` check; `python -m api.auth` generates a key + hash
- `api/request_log.py`: `request_logs` table and SQLAlchemy Core logger
- `api/config.py`: environment settings (see `.env.example`)
- `tests/test_api.py`, `tests/test_request_log.py`

### Setup and run (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env          # then fill in the two required values
python -m api.auth              # prints a new key and its SHA-256 hash
uvicorn api.main:create_app --factory --env-file .env --port 8000
```
The API refuses to start without `TRUTHLAYER_API_KEY_HASHES` and
`TRUTHLAYER_DATABASE_URL` (a PostgreSQL 15 URL,
`postgresql+psycopg://...`). The `request_logs` table is created on startup.
Interactive docs: http://localhost:8000/docs. For the frontend, set
`NEXT_PUBLIC_USE_MOCK=false` in `frontend/.env.local` and paste the key into
its API key field.

### Request
`POST /verify`, header `X-API-Key: <key>`:
```json
{"text": "The Earth orbits the Sun.", "knowledge_base_id": "wikipedia_subset"}
```
`knowledge_base_id` is optional; only the single configured ID
(`TRUTHLAYER_KNOWLEDGE_BASE_ID`) is accepted.

### Response (200)
```json
{
  "status": "ok",
  "message": null,
  "claims": [{
    "claim_id": 1,
    "text": "The Earth orbits the Sun.",
    "span": {"start_char": 0, "end_char": 25},
    "source_sentence": "The Earth orbits the Sun.",
    "source_sentence_span": {"start_char": 0, "end_char": 25},
    "verdict": "Supported",
    "confidence": 0.97,
    "citations": [{"source_id": "KB-001", "text": "The Earth orbits the Sun once approximately every 365.25 days.", "relevance_score": 0.81}]
  }],
  "processing_time_ms": 812
}
```
Spans are half-open character offsets into the request `text`. When no
claims are found: `status: "no_verifiable_claims"`, `message: "No
verifiable claims found"`, `claims: []` (still 200, REQ-4).

### Errors
All errors return `{"detail": "<message>"}`.
- **400**: malformed JSON, missing or non-string `text`, input over 2,000 words, `knowledge_base_id` over 64 characters or unknown.
- **401**: missing or invalid `X-API-Key` (checked before the body is validated, except for unparseable JSON).
- **500**: pipeline failure; internal details are logged server-side only.

### Request log (PostgreSQL)
One row per `/verify` request (including 400/401/500): timestamp (UTC, when
the request arrived), status code, processing time, input word/char counts,
knowledge base ID, claim and per-verdict counts, error type. The submitted
text, claims, evidence, API key and client IP are **not** stored. Rows older
than `TRUTHLAYER_LOG_RETENTION_DAYS` (default 30) are purged on startup. If a
log write fails, the request still succeeds and only the exception type is
logged. Read rows back (newest first) with
`SqlRequestLogger.from_url(url).recent(limit=100)`.

### Tests
```powershell
python -m pytest
```
API tests use fake models and an in-memory logger, so they don't need model
downloads or database credentials. The logger's SQL is unit-tested against
in-memory SQLite. To also run it against real PostgreSQL, point
`TRUTHLAYER_TEST_DATABASE_URL` at a **dedicated** test database. The tests
fail closed: the database name must end in `_test` (e.g. `truthlayer_test`),
must differ from `TRUTHLAYER_DATABASE_URL`, and the URL may not carry
`service`/`dbname`/`options`/`host` query options. After connecting, the
tests check `current_database()` and then run inside a randomly named
`tl_test_*` schema that is dropped afterwards, so no other rows can be purged.

CI (`.github/workflows/backend-tests.yml`) runs `pytest` on Python 3.11 for
pull requests and pushes to `main`, with a throwaway `postgres:15` service
as the `truthlayer_test` database, so the PostgreSQL tests run there too. It
skips installing the ML model packages, so the real-model test is skipped.

### Known gaps
- Confidence is the raw NLI score; calibration (REQ-11) is not implemented.
- No knowledge-base registry or upload (REQ-9 via API); one configured KB file.
- No NLI-failure fallback to similarity-only verification (SRS section 7).
- No batch/evaluation endpoints (REQ-17/18); no HTTPS termination (NFR-8, deployment concern).


## Run with Docker (backend + frontend + PostgreSQL)

`compose.yaml` runs three containers: `db` (PostgreSQL 15), `backend`
(FastAPI, built from `Dockerfile`) and `frontend` (Next.js, built from
`frontend/Dockerfile`). The browser loads the UI from `localhost:3000` and
calls the API directly at `localhost:8000`; the backend reaches the database
by its service name, `db`.

### Prerequisites
- Docker Desktop (Compose v2), with about 4 GB of free disk space.
- Internet access on the first start: the backend downloads its embedding and
  NLI models (~0.6 GB) into the `hf-cache` volume. Later starts reuse them.
- Python on the host, only to generate an API key (any Python 3.11+, no packages).
- Free host ports 3000, 8000 and 5433.

### 1. Configure `.env`
From the repo root (PowerShell):
```powershell
copy .env.example .env
python -m api.auth                 # prints a new API key and its SHA-256 hash
python -c "import secrets; print(secrets.token_urlsafe(24))"   # a database password
```
In `.env`, set:
- `TRUTHLAYER_API_KEY_HASHES=` to the **hash** (never the key). Keep the key
  itself somewhere safe; you will paste it into the UI. For several keys,
  separate the hashes with commas.
- `POSTGRES_PASSWORD=` to the generated password. Use only letters, digits,
  `-` and `_`, since it ends up inside the database URL.

Compose builds `TRUTHLAYER_DATABASE_URL` itself (pointing at `db`), so leave
that line as it is. `.env` is git-ignored; never commit it.

### 2. Start
```powershell
docker compose up -d --build
docker compose ps                  # wait until all three show "(healthy)"
```
On the first start the backend stays `health: starting` for a few minutes
while it downloads the models (`docker compose logs -f backend` shows progress).

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 (paste your API key into the API key field) |
| Backend API | http://localhost:8000/verify |
| API docs | http://localhost:8000/docs |
| PostgreSQL | `localhost:5433` (user/db `truthlayer`, password from `.env`) |

All ports are bound to `127.0.0.1`, so they're reachable only from your machine.

Smoke test (PowerShell; expect 401 without a key, then 200 with one):
```powershell
curl.exe -s -X POST http://localhost:8000/verify -H "Content-Type: application/json" -d '{\"text\": \"The Earth orbits the Sun.\"}'
curl.exe -s -X POST http://localhost:8000/verify -H "Content-Type: application/json" -H "X-API-Key: <your key>" -d '{\"text\": \"The Earth orbits the Sun.\"}'
```

### Everyday commands
```powershell
docker compose logs -f backend     # or frontend / db; all services: docker compose logs -f
docker compose up -d --build       # rebuild and restart after code changes
docker compose build frontend      # rebuild one image
docker compose restart backend     # e.g. after editing .env (use up -d to apply compose.yaml changes)
docker compose down                # stop and remove containers; data and model cache are kept
```
`docker compose down -v` also deletes the `pgdata` (request logs) and
`hf-cache` (models) volumes. Only use it when you want a clean slate.

Request logs in the database:
```powershell
docker compose exec db psql -U truthlayer -d truthlayer -c "SELECT created_at, status_code, processing_time_ms, claim_count FROM request_logs ORDER BY id DESC LIMIT 10"
```

### Troubleshooting
- **`required variable ... is missing a value`**: `.env` is missing, or
  `TRUTHLAYER_API_KEY_HASHES` / `POSTGRES_PASSWORD` is empty. Run compose from
  the repo root, where `.env` lives.
- **Backend restarts with `ConfigError`**: the hash is not a 64-character hex
  SHA-256. Paste the hash printed by `python -m api.auth`, not the key.
- **Backend stuck at `health: starting`**: check `docker compose logs backend`.
  The first start needs internet access to huggingface.co.
- **`password authentication failed`**: the database volume was created with an
  older `POSTGRES_PASSWORD`; PostgreSQL only reads it on first initialisation.
  Restore the old password in `.env`, or delete the data with `docker compose down -v`.
- **Port already in use**: stop whatever uses 3000/8000, or change
  `POSTGRES_HOST_PORT` in `.env` for the database. The API and UI ports are fixed
  in `compose.yaml` because the UI's API URL is built into the frontend image.
- **UI shows "Cannot reach the TruthLayer server"**: check that the backend is healthy.
  If you serve the UI from a different origin, add it to `TRUTHLAYER_CORS_ORIGINS`.
- **UI shows 401**: the key pasted in the UI doesn't match any configured hash.
- **The UI still uses old settings**: `NEXT_PUBLIC_*` values are built into the
  frontend image; rebuild with `docker compose build frontend`.


## Running the tests

```powershell
python -m pip install -r requirements.txt -r requirements-calibration.txt
python -m pytest -rs                      # all backend tests (fake models, no downloads)
python -m pytest tests/test_integration_pipeline.py -v   # extraction -> retrieval -> NLI -> API
python -m pytest tests/test_security.py -v               # auth, invalid input, no sensitive logging
cd frontend; npm test                     # frontend tests
```

The 5 PostgreSQL tests in `tests/test_request_log.py` are skipped unless a dedicated
test database is configured (CI runs them against a throwaway `postgres:15`):

```powershell
$env:TRUTHLAYER_TEST_DATABASE_URL="postgresql+psycopg://<user>:<password>@localhost:5432/truthlayer_test"
python -m pytest tests/test_request_log.py -rs
```

## Evaluation and performance scripts (real models; first run downloads them)

```powershell
python scripts/evaluate_verification.py   # precision/recall/F1 -> results/verification_eval.{json,md}
$env:TRUTHLAYER_TEST_KEY="<plaintext api key>"
python scripts/perf_test.py               # NFR-1/2/3 -> results/performance_results.json (API must be running)
```