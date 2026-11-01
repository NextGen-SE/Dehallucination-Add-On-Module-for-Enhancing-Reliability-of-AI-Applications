
"""Run confidence calibration evaluation for TruthLayer.

Run from the repository root:
    python scripts/evaluate_calibration.py

Uses the real embedding and NLI adapters. The first run may download
Hugging Face models and therefore needs internet access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    """Create a source-separated split when possible."""

    labels = frame["correct"].astype(int).to_numpy()
    groups = frame["source_id"].fillna("NO_SOURCE").astype(str).to_numpy()

    if len(set(labels)) != 2:
        raise ValueError(
            "Calibration requires both correct (1) and incorrect (0) predictions."
        )

    if min(pd.Series(labels).value_counts()) < 2:
        raise ValueError(
            "At least two examples from each correctness class are required."
        )

    if len(set(groups)) >= 3:
        splitter = GroupShuffleSplit(
            n_splits=500,
            test_size=0.30,
            random_state=42,
        )

        for train_idx, test_idx in splitter.split(frame, labels, groups):
            if (
                len(set(labels[train_idx])) == 2
                and len(set(labels[test_idx])) == 2
                and len(test_idx) >= 2
            ):
                return train_idx, test_idx, "GroupShuffleSplit by source_id"

    indices = list(range(len(frame)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    return (
        train_idx,
        test_idx,
        "Stratified row-level split (source separation unavailable)",
    )


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

    dataset = pd.read_csv(DATA_PATH)

    required = {"claim_text", "expected_verdict"}
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    if dataset["claim_text"].isna().any() or dataset["expected_verdict"].isna().any():
        raise ValueError("claim_text and expected_verdict cannot contain missing values.")

    print(f"Loaded {len(dataset)} evaluation examples.")
    print("Loading knowledge base and verifier...")

    kb = load_knowledge_base(KB_PATH)
    verifier = EvidenceVerifier(knowledge_base=kb)

    records = []

    for index, row in enumerate(dataset.to_dict(orient="records"), start=1):
        result = verifier.verify_claim(str(row["claim_text"]))

        expected = str(row["expected_verdict"]).strip()
        predicted = result.verdict

        eligible = (
            expected in BINARY_VERDICTS
            and predicted in BINARY_VERDICTS
        )

        correct = predicted == expected

        print(
            f"Claim: {str(row['claim_text'])[:65]:65} | "
            f"Expected: {expected:12} | "
            f"Predicted: {predicted:12} | "
            f"Confidence: {result.confidence:.3f} | "
            f"Correct: {correct}"
        )

        records.append({
            "claim_text": str(row["claim_text"]),
            "source_id": row.get("source_id"),
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "raw_confidence": float(result.confidence),
            "correct": int(correct),
            "included_in_calibration": bool(eligible),
            "exclusion_reason": (
                None if eligible else
                "Expected or predicted verdict is Unverifiable; "
                "its confidence may come from a different scoring path."
            ),
        })

        if index % 5 == 0 or index == len(dataset):
            print(f"Processed {index}/{len(dataset)} examples.")

    all_results = pd.DataFrame(records)

    eligible = all_results[
        all_results["included_in_calibration"]
    ].copy()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save every prediction before attempting calibration.
    predictions_path = RESULTS_DIR / "confidence_evaluation_predictions.csv"
    all_results.to_csv(predictions_path, index=False)

    print("\n--- Overall Evaluation ---")
    print("Total examples:", len(all_results))
    print("Overall correct:", int(all_results["correct"].sum()))
    print("Overall accuracy:", round(all_results["correct"].mean(), 4))

    print("\nPredicted verdict distribution:")
    print(all_results["predicted_verdict"].value_counts().to_dict())

    print("\n--- Unverifiable Evaluation ---")
    unverifiable = all_results[
        all_results["expected_verdict"] == "Unverifiable"
    ]

    unverifiable_correct = int(
        (
            unverifiable["predicted_verdict"] == "Unverifiable"
        ).sum()
    )

    print("Unverifiable examples:", len(unverifiable))
    print("Correctly predicted Unverifiable:", unverifiable_correct)

    if len(unverifiable):
        print(
            "Unverifiable accuracy:",
            round(unverifiable_correct / len(unverifiable), 4),
        )

    print("\n--- Binary Calibration Evaluation ---")
    print("Eligible examples:", len(eligible))
    print("Excluded examples:", len(all_results) - len(eligible))
    print("Correctness distribution:", eligible["correct"].value_counts().to_dict())

    if len(eligible) < 8:
        print("\nCalibration skipped: fewer than 8 eligible examples.")
        print(f"Prediction diagnostics saved: {predictions_path}")
        return 0

    # Calibration needs both correct and incorrect predictions.
    if eligible["correct"].nunique() != 2:
        print(
            "\nCalibration skipped: the dataset does not contain "
            "both correct and incorrect binary predictions."
        )
        print("Do not artificially change expected labels to force calibration.")
        print(f"Prediction diagnostics saved: {predictions_path}")
        return 0

    if min(eligible["correct"].value_counts()) < 2:
        print(
            "\nCalibration skipped: at least two examples from each "
            "correctness class are required."
        )
        print(f"Prediction diagnostics saved: {predictions_path}")
        return 0

    print("\nSource distribution:")
    print(
        eligible.groupby("source_id", dropna=False)["correct"]
        .value_counts()
        .to_string()
    )

    train_idx, test_idx, split_strategy = make_grouped_split(eligible)

    train = eligible.iloc[train_idx].copy()
    test = eligible.iloc[test_idx].copy()

    print("\nSelected split:", split_strategy)
    print("Calibration examples:", len(train))
    print("Held-out examples:", len(test))

    if train["correct"].nunique() != 2 or test["correct"].nunique() != 2:
        print("\nCalibration skipped: split does not contain both classes.")
        print(f"Prediction diagnostics saved: {predictions_path}")
        return 0

    calibrator = ConfidenceCalibrator().fit(
        train["raw_confidence"].to_numpy(),
        train["correct"].to_numpy(),
    )

    test["calibrated_confidence"] = calibrator.predict_calibrated(
        test["raw_confidence"].to_numpy()
    )

    raw_metrics = calibration_metrics(
        test["raw_confidence"].to_numpy(),
        test["correct"].to_numpy(),
    )

    calibrated_metrics = calibration_metrics(
        test["calibrated_confidence"].to_numpy(),
        test["correct"].to_numpy(),
    )

    output = {
        "method": "Platt scaling (logistic regression)",
        "target": (
            "Probability that the predicted Supported/Contradicted "
            "verdict is correct"
        ),
        "split": {
            "strategy": split_strategy,
            "calibration_examples": int(len(train)),
            "held_out_examples": int(len(test)),
            "calibration_source_groups": sorted(
                train["source_id"].fillna("NO_SOURCE")
                .astype(str).unique().tolist()
            ),
            "held_out_source_groups": sorted(
                test["source_id"].fillna("NO_SOURCE")
                .astype(str).unique().tolist()
            ),
            "source_overlap_possible": (
                "Stratified row-level" in split_strategy
            ),
        },
        "dataset_summary": {
            "total_examples": int(len(all_results)),
            "eligible_examples": int(len(eligible)),
            "excluded_examples": int(len(all_results) - len(eligible)),
            "correct_predictions": int((eligible["correct"] == 1).sum()),
            "incorrect_predictions": int((eligible["correct"] == 0).sum()),
        },
        "calibrator_parameters": calibrator.parameters,
        "held_out_metrics": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
        "held_out_predictions": test[
            [
                "claim_text",
                "source_id",
                "expected_verdict",
                "predicted_verdict",
                "raw_confidence",
                "calibrated_confidence",
                "correct",
            ]
        ].to_dict(orient="records"),
        "all_model_predictions": records,
        "limitations": [
            "The evaluation dataset is small and requires independent label review.",
            "Only Supported/Contradicted predictions are calibrated.",
            "Unverifiable predictions are excluded because their confidence uses a different scoring path.",
            "Source-separated splitting is preferred.",
            "Results are exploratory and do not establish production-grade calibration.",
            "The calibrator estimates prediction correctness; it does not change verdicts.",
        ],
    }

    json_path = RESULTS_DIR / "calibration_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "TRUTHLAYER CONFIDENCE CALIBRATION REPORT",
        "=" * 44,
        f"Method: {output['method']}",
        f"Dataset examples: {len(all_results)}",
        f"Eligible binary-verdict examples: {len(eligible)}",
        f"Correct predictions: {output['dataset_summary']['correct_predictions']}",
        f"Incorrect predictions: {output['dataset_summary']['incorrect_predictions']}",
        f"Calibration examples: {len(train)}",
        f"Held-out examples: {len(test)}",
        f"Split strategy: {split_strategy}",
        "",
        "Held-out comparison (lower Brier score / ECE is better):",
    ]

    for label, metrics in (
        ("Raw", raw_metrics),
        ("Calibrated", calibrated_metrics),
    ):
        report_lines.extend([
            f"{label}:",
            f"  Brier score:       {metrics['brier_score']:.4f}",
            f"  ECE:               {metrics['ece']:.4f}",
            f"  Mean confidence:   {metrics['mean_confidence']:.4f}",
            f"  Observed accuracy: {metrics['observed_accuracy']:.4f}",
        ])

    report_lines.extend(
        ["", "Limitations:"]
        + [f"- {item}" for item in output["limitations"]]
    )

    report_path = RESULTS_DIR / "calibration_report.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n--- Calibration Evaluation Completed ---")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")
    print("\nHeld-out metrics:")
    print(json.dumps(output["held_out_metrics"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
