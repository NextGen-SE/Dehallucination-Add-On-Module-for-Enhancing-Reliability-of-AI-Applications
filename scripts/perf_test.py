"""Performance checks for Issue #27 (SRS NFR-1, NFR-2, NFR-3).

1. Retrieval time per claim      (NFR-3, target <= 500 ms)   -- in-process, real models
2. /verify latency, ~200 words   (NFR-1, target <= 5 s)      -- needs the API running
3. 10 concurrent /verify calls   (NFR-2, all must succeed)   -- needs the API running

Start the API first (see README), then from the repository root:
    $env:TRUTHLAYER_TEST_KEY = "<your PLAINTEXT api key>"      # PowerShell
    python scripts/perf_test.py
    python scripts/perf_test.py --skip-retrieval          # API checks only
    python scripts/perf_test.py --skip-api                # retrieval only

Writes results/performance_results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NFR1_SECONDS = 5.0
NFR3_SECONDS = 0.5
CONCURRENT_REQUESTS = 10

SAMPLE_CLAIMS = [
    "The Earth orbits the Sun once every 365 days.", "Water is made of hydrogen and oxygen.",
    "Bengaluru is the capital of Karnataka.", "The Moon is made of cheese.",
    "Mount Everest is in the Himalayas.", "The human heart has four chambers.",
    "Python was first released in 1991.", "The speed of light is about 300000 kilometers per second.",
    "India became independent in 1947.", "The Pacific Ocean is the largest ocean.",
    "DNA carries genetic instructions.", "The French Revolution began in 1789.",
    "The Amazon River flows through Brazil.", "Tokyo is the capital of Japan.",
    "Photosynthesis needs sunlight.", "The Great Wall was built by several dynasties.",
]

# Repeats factual sentences to reach ~195 words (NFR-1 is defined for up to 200 words).
_PARAGRAPH = (
    "The Earth orbits the Sun once approximately every year. Water is made of hydrogen and oxygen atoms. "
    "Bengaluru is the capital city of the Indian state of Karnataka. Mount Everest is the tallest mountain above sea level. "
    "The human heart has four chambers called atria and ventricles. Python is an interpreted programming language released in 1991. "
)
TEXT_200_WORDS = (_PARAGRAPH * 3 + " ".join(_PARAGRAPH.split(". ")[:3]) + ".").strip()


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values), "p95": p95(values), "max": max(values)}


def measure_retrieval(repeats: int = 3) -> dict:
    from evidence_verification import EvidenceVerifier, load_knowledge_base

    verifier = EvidenceVerifier(load_knowledge_base(ROOT / "knowledge_base.json"))
    verifier.retrieve("warm up")  # exclude one-time model warm-up from timings
    timings = []
    for _ in range(repeats):
        for claim in SAMPLE_CLAIMS:
            started = time.perf_counter()
            verifier.retrieve(claim)
            timings.append(time.perf_counter() - started)
    stats = summarize(timings)
    stats["target_seconds"] = NFR3_SECONDS
    stats["passed"] = stats["p95"] <= NFR3_SECONDS
    return stats


def _post(client: httpx.Client, url: str, key: str, text: str) -> tuple[int, float, dict]:
    started = time.perf_counter()
    response = client.post(url, json={"text": text}, headers={"X-API-Key": key}, timeout=300)
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, elapsed, body


def measure_single_latency(base_url: str, key: str, runs: int = 5) -> dict:
    timings, server_ms, claim_counts = [], [], []
    with httpx.Client() as client:
        _post(client, f"{base_url}/verify", key, TEXT_200_WORDS)  # warm-up
        for _ in range(runs):
            status, elapsed, body = _post(client, f"{base_url}/verify", key, TEXT_200_WORDS)
            if status != 200:
                raise RuntimeError(f"/verify returned {status}: {body}")
            timings.append(elapsed)
            server_ms.append(body["processing_time_ms"])
            claim_counts.append(len(body["claims"]))
    stats = summarize(timings)
    stats.update(words=len(TEXT_200_WORDS.split()), claims_per_request=claim_counts[0],
                 server_processing_ms_mean=statistics.mean(server_ms),
                 target_seconds=NFR1_SECONDS, passed=stats["max"] <= NFR1_SECONDS)
    return stats


def measure_concurrency(base_url: str, key: str, n: int = CONCURRENT_REQUESTS) -> dict:
    def call(_):
        with httpx.Client() as client:
            status, elapsed, _body = _post(client, f"{base_url}/verify", key, TEXT_200_WORDS)
            return status, elapsed

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(call, range(n)))
    wall = time.perf_counter() - started
    statuses = [s for s, _ in results]
    latencies = [t for _, t in results]
    return {"requests": n, "statuses": statuses, "all_succeeded": all(s == 200 for s in statuses),
            "wall_seconds": wall, "slowest_seconds": max(latencies), "fastest_seconds": min(latencies),
            "note": "Pipeline serialises model calls with a lock (api/pipeline.py), so requests queue; "
                    "NFR-2 allows queuing but the slowest request includes waiting time.",
            "passed": all(s == 200 for s in statuses)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    results: dict = {"text_words": len(TEXT_200_WORDS.split())}
    if not args.skip_retrieval:
        print("Measuring retrieval time (NFR-3)...")
        results["nfr3_retrieval"] = measure_retrieval()
    if not args.skip_api:
        key = os.environ.get("TRUTHLAYER_TEST_KEY")
        if not key:
            print("Set TRUTHLAYER_TEST_KEY to your plaintext API key (or use --skip-api).")
            return 2
        print("Measuring single-request latency (NFR-1)...")
        results["nfr1_latency"] = measure_single_latency(args.base_url, key)
        print(f"Measuring {CONCURRENT_REQUESTS} concurrent requests (NFR-2)...")
        results["nfr2_concurrency"] = measure_concurrency(args.base_url, key)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "performance_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(v.get("passed", True) for v in results.values() if isinstance(v, dict)) else 1


if __name__ == "__main__":
    raise SystemExit(main())