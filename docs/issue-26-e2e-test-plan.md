# Issue #26 — End-to-End Test Plan and Execution Record

**Objective:** Validate the TruthLayer workflow from text submission through API verification to displayed dashboard results.

**Important:** This file is a test plan and execution record template. A test is not marked PASS until it has actually been run in the target environment. Automated test code is provided, but local execution is still required.

## Automated test coverage

| Test ID | Workflow / requirement | Automated coverage | Initial status |
|---|---|---|---|
| E2E-01 | Submit text and receive verification result | `tests/test_issue26_e2e.py::test_dashboard_workflow_api_returns_all_verdicts_and_citations` | NOT RUN |
| E2E-02 | Supported, Contradicted and Unverifiable verdicts | Same backend test; `frontend/__tests__/page.test.js` full-results test | NOT RUN |
| E2E-03 | Missing evidence produces Unverifiable without citations | `tests/test_issue26_e2e.py::test_missing_evidence_is_unverifiable_and_has_no_citation` | NOT RUN |
| E2E-04 | Invalid/malformed input produces understandable HTTP 400 | `tests/test_api.py` bad-request tests; Issue #18 validation tests | NOT RUN |
| E2E-05 | Missing/invalid API key is rejected | `tests/test_api.py` authentication tests; Issue #26 workflow test | NOT RUN |
| E2E-06 | Evidence citations are returned and displayed | `tests/test_api.py` citation tests; frontend results test | NOT RUN |
| E2E-07 | Batch verification workflow | Depends on Issue #22 implementation | BLOCKED — feature not present in supplied project snapshot |
| E2E-08 | Downloadable batch report | Depends on Issue #23 implementation | BLOCKED — feature not present in supplied project snapshot |
| E2E-09 | UI displays API errors without crashing | `frontend/__tests__/page.test.js` error-banner test | NOT RUN |
| E2E-10 | UI result values match API response | API schema tests plus frontend normalization/results tests | NOT RUN |
| E2E-11 | Request metadata is logged | `tests/test_issue26_e2e.py::test_request_metadata_is_logged_without_submitted_text` and `tests/test_api.py` logging tests | NOT RUN |
| E2E-12 | No-checkable-claims response is displayed as a normal result | `tests/test_issue26_e2e.py::test_no_checkable_claims_returns_successful_empty_result` and frontend page test | NOT RUN |

## Test environment

- Backend: Python 3.11 recommended, FastAPI, pytest, TestClient.
- Frontend: Node.js, Next.js, Jest, React Testing Library.
- API: FastAPI `POST /verify`.
- Test data: deterministic in-memory knowledge base and fake embedder/NLI adapters for repeatable tests.
- PostgreSQL: required only for PostgreSQL-specific logger tests and the full CI job; use a dedicated database ending in `_test`.

## Run tests

Backend, from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
```

Frontend:

```powershell
cd frontend
npm ci
npm test -- --runInBand
```

For a live dashboard-to-backend check, configure `frontend/.env.local` with `NEXT_PUBLIC_API_URL=http://localhost:8000` and `NEXT_PUBLIC_USE_MOCK=false`, start the API and dashboard, and submit the same sample text used by the backend tests.

## Manual end-to-end scenarios

1. Start PostgreSQL and configure the backend `.env` values.
2. Start the API using `uvicorn api.main:create_app --factory --env-file .env --port 8000`.
3. Start the dashboard with `npm run dev` from `frontend/`.
4. Enter a factual statement that the configured knowledge base supports. Confirm the verdict and citation.
5. Enter a factual statement that contradicts a knowledge-base passage. Confirm the Contradicted verdict and evidence.
6. Enter a claim for which the knowledge base has no relevant passage. Confirm Unverifiable and no fabricated citation.
7. Submit empty, malformed, and over-2,000-word input. Confirm clear validation feedback.
8. Remove or mistype the API key. Confirm an authentication error is shown and the page remains usable.
9. Compare the displayed claim text, verdict, confidence, and citations with the JSON returned by `POST /verify`.
10. After Issues #22 and #23 are integrated, execute batch submission and report download scenarios and attach the generated artifacts.

## Execution record

Complete this table after running tests in the team's environment. Record actual output, date, tester, and defect IDs. Do not change NOT RUN to PASS based only on the presence of test code.

| Date | Tester | Command / test IDs | Passed | Failed | Blocked | Defect IDs / notes |
|---|---|---|---:|---:|---:|---|
| Pending | Pending | Backend pytest | — | — | — | Not executed in this package preparation |
| Pending | Pending | Frontend Jest | — | — | — | Not executed in this package preparation |
| Pending | Pending | Manual dashboard + API | — | — | — | Requires local environment and configured credentials |

## Known unresolved items

- Batch verification and batch report download cannot be end-to-end tested until Issues #22 and #23 are merged.
- A real deployment HTTPS/TLS check must be performed in the deployment environment.
- Real model quality and performance are not measured by fake-model automated tests. Run the held-out benchmark and performance/security checks separately and record the results.
