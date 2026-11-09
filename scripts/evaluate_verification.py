"""Evaluate the verifier: precision, recall, F1 per verdict (Issue #21).

Run from the repository root (uses the REAL models; first run downloads them):
    python scripts/evaluate_verification.py
    python scripts/evaluate_verification.py --data data/eval_claims.csv

Outputs:
    results/verification_eval.json   machine-readable metrics
    results/verification_eval.md     readable tables for the report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from evidence_verification import (  # noqa: E402
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    EvidenceVerifier,
    load_knowledge_base,
)

LABELS = ["Supported", "Contradicted", "Unverifiable"]
REQ_19_ACCURACY_TARGET = 0.80


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for number, row in enumerate(rows, start=2):
        if row["expected_verdict"] not in LABELS:
            raise ValueError(f"{path}:{number}: unknown label {row['expected_verdict']!r}")
    return rows


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        "sample_count": len(y_true),
        "accuracy": accuracy,
        "meets_req_19_accuracy_target": accuracy >= REQ_19_ACCURACY_TARGET,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "macro_avg": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f)},
        "confusion_matrix": {
            "labels": LABELS,
            "rows_are": "expected",
            "columns_are": "predicted",
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        # NFR-4: insufficient evidence must never be presented as Supported.
        "nfr4_unverifiable_predicted_supported": sum(
            t == "Unverifiable" and p == "Supported" for t, p in zip(y_true, y_pred)
        ),
    }


def to_markdown(metrics: dict, config: dict) -> str:
    lines = [
        "| Verdict | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label, m in metrics["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    macro = metrics["macro_avg"]
    lines.append(f"| **Macro average** | {macro['precision']:.3f} | {macro['recall']:.3f} | {macro['f1']:.3f} | {metrics['sample_count']} |")
    lines += ["", f"Overall accuracy: **{metrics['accuracy']:.3f}** (REQ-19 target: {REQ_19_ACCURACY_TARGET:.2f}, "
              f"{'met' if metrics['meets_req_19_accuracy_target'] else 'NOT met'})",
              f"Unverifiable claims predicted Supported (NFR-4, must be 0): **{metrics['nfr4_unverifiable_predicted_supported']}**",
              "", "Confusion matrix (rows = expected, columns = predicted):", "",
              "| expected \\ predicted | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    for label, row in zip(LABELS, metrics["confusion_matrix"]["matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", f"Config: top_k={config['top_k']}, relevance_threshold={config['relevance_threshold']}, "
              f"knowledge base passages={config['kb_size']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "eval_claims.csv")
    parser.add_argument("--kb", type=Path, default=ROOT / "knowledge_base.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    rows = load_rows(args.data)
    kb = load_knowledge_base(args.kb)
    verifier = EvidenceVerifier(kb)  # real models

    started = time.perf_counter()
    y_true = [r["expected_verdict"] for r in rows]
    y_pred, misses = [], []
    for row in rows:
        result = verifier.verify_claim(row["claim_text"])
        y_pred.append(result.verdict)
        if result.verdict != row["expected_verdict"]:
            misses.append({
                "claim_text": row["claim_text"],
                "expected": row["expected_verdict"],
                "predicted": result.verdict,
                "confidence": result.confidence,
                "evidence_source": result.evidence[0].source_id if result.evidence else None,
            })
    elapsed = time.perf_counter() - started

    config = {"top_k": verifier.top_k, "relevance_threshold": verifier.relevance_threshold,
              "kb_size": len(kb), "dataset": str(args.data.name)}
    metrics = compute_metrics(y_true, y_pred)
    metrics["config"] = config
    metrics["seconds_total"] = round(elapsed, 2)
    metrics["misclassified"] = misses

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "verification_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    markdown = to_markdown(metrics, config)
    (args.out_dir / "verification_eval.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"{len(misses)} misclassified claims listed in results/verification_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())