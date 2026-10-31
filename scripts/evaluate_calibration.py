"""Run confidence calibration evaluation for TruthLayer.

Run from the repository root:
    python scripts/evaluate_calibration.py

This script uses the real embedding and NLI adapters. The first run may
download Hugging Face models and therefore needs internet access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import ConfidenceCalibrator, calibration_metrics
from evidence_verification import (
    EvidenceVerifier,
    VERDICT_CONTRADICTED,
    VERDICT_SUPPORTED,
    load_knowledge_base,
)

DATA_PATH = ROOT / "data" / "confidence_evaluation.csv"
KB_PATH = ROOT / "knowledge_base.json"
RESULTS_DIR = ROOT / "results"
BINARY_VERDICTS = {VERDICT_SUPPORTED, VERDICT_CONTRADICTED}


def make_grouped_split(frame: pd.DataFrame):
    """Find a deterministic source-group split with both target classes."""
    groups = frame["source_id"].fillna("NO_SOURCE").astype(str).to_numpy()
    labels = frame["correct"].astype(int).to_numpy()

    if len(set(groups)) < 3:
        raise ValueError("At least three distinct source groups are needed for a held-out split.")

    splitter = GroupShuffleSplit(n_splits=100, test_size=0.30, random_state=42)
    for train_idx, test_idx in splitter.split(frame, labels, groups):
        if len(set(labels[train_idx])) == 2 and len(test_idx) >= 2:
            return train_idx, test_idx

    raise ValueError(
        "Could not create a source-separated split with both correctness classes "
        "in the calibration subset. Review/add labeled examples before calibration."
    )


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Keep your existing data/confidence_evaluation.csv in the repository."
        )
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    required = {"claim_text", "expected_verdict"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if dataset["claim_text"].isna().any() or dataset["expected_verdict"].isna().any():
        raise ValueError("claim_text and expected_verdict cannot contain missing values.")

    kb = load_knowledge_base(KB_PATH)
    verifier = EvidenceVerifier(knowledge_base=kb)

    records = []
    for row in dataset.to_dict(orient="records"):
        result = verifier.verify_claim(str(row["claim_text"]))
        expected = str(row["expected_verdict"]).strip()
        predicted = result.verdict
        eligible = expected in BINARY_VERDICTS and predicted in BINARY_VERDICTS

        records.append({
            "claim_text": str(row["claim_text"]),
            "source_id": row.get("source_id"),
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "raw_confidence": float(result.confidence),
            "correct": int(predicted == expected),
            "included_in_calibration": bool(eligible),
            "exclusion_reason": (
                None if eligible else
                "Expected or predicted verdict is Unverifiable; its confidence "
                "may come from a different scoring path."
            ),
        })

    all_results = pd.DataFrame(records)
    eligible = all_results[all_results["included_in_calibration"]].copy()
    if len(eligible) < 8:
        raise ValueError(
            f"Only {len(eligible)} eligible binary-verdict predictions. "
            "At least 8 are required for this demonstration; add/review examples."
        )

    # Calibration and test groups are separated by source_id to reduce leakage
    # from multiple claims grounded in the same knowledge-base passage.
    train_idx, test_idx = make_grouped_split(eligible)
    train = eligible.iloc[train_idx]
    test = eligible.iloc[test_idx]

    calibrator = ConfidenceCalibrator().fit(
        train["raw_confidence"].to_numpy(),
        train["correct"].to_numpy(),
    )
    test = test.copy()
    test["calibrated_confidence"] = calibrator.predict_calibrated(
        test["raw_confidence"].to_numpy()
    )

    raw_metrics = calibration_metrics(
        test["raw_confidence"].to_numpy(), test["correct"].to_numpy()
    )
    calibrated_metrics = calibration_metrics(
        test["calibrated_confidence"].to_numpy(), test["correct"].to_numpy()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "Platt scaling (logistic regression)",
        "target": "Probability that the predicted Supported/Contradicted verdict is correct",
        "split": {
            "strategy": "GroupShuffleSplit by source_id",
            "calibration_examples": int(len(train)),
            "held_out_examples": int(len(test)),
            "calibration_source_groups": sorted(train["source_id"].fillna("NO_SOURCE").astype(str).unique().tolist()),
            "held_out_source_groups": sorted(test["source_id"].fillna("NO_SOURCE").astype(str).unique().tolist()),
        },
        "dataset_summary": {
            "total_examples": int(len(all_results)),
            "eligible_examples": int(len(eligible)),
            "excluded_examples": int(len(all_results) - len(eligible)),
            "excluded_unverifiable_or_mixed_score_examples": int(len(all_results) - len(eligible)),
        },
        "calibrator_parameters": calibrator.parameters,
        "held_out_metrics": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
        "held_out_predictions": test[
            ["claim_text", "source_id", "expected_verdict", "predicted_verdict",
             "raw_confidence", "calibrated_confidence", "correct"]
        ].to_dict(orient="records"),
        "all_model_predictions": records,
        "limitations": [
            "The evaluation dataset is small and its labels/evidence alignment require human review.",
            "Only Supported/Contradicted predictions are calibrated; Unverifiable predictions are excluded because their confidence can come from a different scoring path.",
            "The held-out set is source-separated, but results remain exploratory and are not evidence of production-grade calibration.",
            "The calibrator estimates correctness of the predicted verdict; it does not change the verdict itself.",
        ],
    }

    json_path = RESULTS_DIR / "calibration_results.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "TRUTHLAYER CONFIDENCE CALIBRATION REPORT",
        "=" * 44,
        f"Method: {output['method']}",
        f"Dataset examples: {len(all_results)}",
        f"Eligible binary-verdict examples: {len(eligible)}",
        f"Calibration examples: {len(train)}",
        f"Held-out examples: {len(test)}",
        "",
        "Held-out comparison (lower Brier score / ECE is better):",
    ]
    for label, metrics in (("Raw", raw_metrics), ("Calibrated", calibrated_metrics)):
        report_lines.extend([
            f"{label}:",
            f"  Brier score:       {metrics['brier_score']:.4f}",
            f"  ECE:               {metrics['ece']:.4f}",
            f"  Mean confidence:   {metrics['mean_confidence']:.4f}",
            f"  Observed accuracy: {metrics['observed_accuracy']:.4f}",
        ])
    report_lines.extend(["", "Limitations:"] + [f"- {item}" for item in output["limitations"]])
    (RESULTS_DIR / "calibration_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Saved: {json_path}")
    print(f"Saved: {RESULTS_DIR / 'calibration_report.txt'}")
    print(json.dumps(output["held_out_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
