# Performance and Security Test Report (Issue #27)

Hardware: Intel Core Ultra 5 125H, 16 GB RAM (15.6 GB usable), no dedicated GPU (integrated Intel Arc),
Windows 11 Home (version 26H2, build 26300.9818). Models: real (all-MiniLM-L6-v2, nli-deberta-v3-small).
Date: 10 November 2026. Commit: eac1d82. Knowledge base: 15 passages.

## Performance (`python scripts/perf_test.py`)
| Requirement | Target | Measured | Result |
|---|---|---|---|
| NFR-3 retrieval per claim | <= 500 ms | mean 14.9 ms / p95 25.4 ms / max 27.2 ms (48 timings) | Pass |
| NFR-1 /verify, 196 words (20 claims) | <= 5 s | mean 1.39 s / max 1.49 s (5 runs, after warm-up) | Pass |
| NFR-2 10 concurrent requests | all succeed | 10/10 returned 200; wall time 14.96 s; slowest 14.10 s; fastest 3.09 s | Pass |

Observations:
- A single 196-word request takes about 1.4 s; NLI verification dominates, retrieval is a small part.
- `api/pipeline.py` serialises model calls with a lock, so concurrent requests queue. The slowest of 10
  concurrent requests took about 10 times a single request (14.1 s vs 1.39 s). The request log shows the same
  pattern: `processing_time_ms` of the concurrent requests rose in steps of about 1.1 s (8.6 s, 9.8 s, 10.9 s, 12.1 s).
  All requests completed successfully, which is what NFR-2 requires, but queued users waited well above the 5 s of NFR-1.

## Security (`python -m pytest tests/test_security.py -v`: 21 tests)
| Requirement | Check | Result |
|---|---|---|
| REQ-13 / NFR-6 | Missing, empty, wrong key -> 401; keys stored as SHA-256 hashes | <Pass> |
| REQ-15 | Bad payloads / invalid JSON -> 400, server keeps serving | <Pass> |
| NFR-7 | Submitted text and API key absent from logs and request_logs table | <Pass> |
| 500 handling | No internal details or text in the error response | <Pass> |
| NFR-4 | Unrelated claim never Supported (also 0 violations in the 150-claim evaluation) | <Pass> |
| NFR-8 | HTTPS/TLS | Not implemented in the app; a deployment concern (needs a reverse proxy) |

Manual check (NFR-7): sent a request containing "ZEBRA-SECRET-123". The newest `request_logs` row stored
only metadata (status 200, 9 words, 55 characters, 2 claims, 1 Supported, 1 Unverifiable, processing time 124 ms).
The sentinel text was absent from the table and from the API console output - confirmed.

## Failures and limitations
- Under 10 simultaneous requests the last user waits about 14 s because model calls are serialised by a lock.
  Possible mitigations (not implemented): multiple worker processes, batching NLI calls, caching repeated claims.
- Measured on one development laptop, a 15-passage knowledge base, and a single 196-word sample text;
  a larger knowledge base would increase retrieval time.
- HTTPS (NFR-8) is not provided by the application itself.
- Request logging was verified locally against SQLite (`sqlite:///truthlayer.db`) because PostgreSQL was not
  installed on the test machine. The SRS (REQ-14) specifies PostgreSQL; the PostgreSQL-specific tests in
  `tests/test_request_log.py` were skipped locally and <are run in CI against a PostgreSQL service / were not run in this environment>.
- The plaintext API key was only held in an environment variable during the test run and is not stored.