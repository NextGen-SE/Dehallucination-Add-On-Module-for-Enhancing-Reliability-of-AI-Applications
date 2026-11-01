"""Platt-style calibration for verification confidence scores.

The calibrator estimates P(prediction is correct | raw confidence).
It does not change the predicted verdict. Fit it only on a calibration
subset, then evaluate on data that was not used during fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ConfidenceCalibrator:
    """Map a model's raw confidence to an estimated probability of correctness."""

    def __init__(self, random_state: int = 42) -> None:
        self._model = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            random_state=random_state,
        )
        self._is_fitted = False

    @staticmethod
    def _validate_confidences(raw_confidences) -> np.ndarray:
        values = np.asarray(raw_confidences, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("At least one confidence score is required.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Confidence scores must be finite numbers.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Confidence scores must be between 0 and 1.")
        return values.reshape(-1, 1)

    def fit(self, raw_confidences, correctness_labels) -> "ConfidenceCalibrator":
        """Fit using raw scores and binary labels (1=correct, 0=incorrect)."""
        x = self._validate_confidences(raw_confidences)
        y = np.asarray(correctness_labels, dtype=int).reshape(-1)

        if len(x) != len(y):
            raise ValueError("Confidence scores and labels must have equal length.")
        if len(y) < 4:
            raise ValueError("At least four examples are required to fit calibration.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Correctness labels must contain only 0 or 1.")
        if np.unique(y).size != 2:
            raise ValueError("Calibration data must contain both correct and incorrect predictions.")

        self._model.fit(x, y)
        self._is_fitted = True
        return self

    def predict_calibrated(self, raw_confidences) -> np.ndarray:
        """Return calibrated probabilities that predictions are correct."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_calibrated().")
        x = self._validate_confidences(raw_confidences)
        probabilities = self._model.predict_proba(x)
        class_one_index = list(self._model.classes_).index(1)
        return probabilities[:, class_one_index]

    @property
    def parameters(self) -> dict:
        """Return model parameters for transparent reporting."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before reading calibration parameters.")
        return {
            "method": "Platt scaling (logistic regression)",
            "coefficient": float(self._model.coef_[0][0]),
            "intercept": float(self._model.intercept_[0]),
        }


def expected_calibration_error(
    confidences, correctness_labels, n_bins: int = 10
) -> float:
    """Compute equal-width expected calibration error (ECE)."""
    scores = np.asarray(confidences, dtype=float).reshape(-1)
    labels = np.asarray(correctness_labels, dtype=float).reshape(-1)

    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError("Scores and labels must be non-empty and have equal length.")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and between 0 and 1.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Correctness labels must contain only 0 or 1.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        # Include the right edge in the final bin.
        if index == n_bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(scores[mask])) - float(np.mean(labels[mask]))
            )
    return float(ece)


def calibration_metrics(confidences, correctness_labels) -> dict:
    """Return Brier score, ECE, mean confidence and observed accuracy."""
    scores = np.asarray(confidences, dtype=float).reshape(-1)
    labels = np.asarray(correctness_labels, dtype=float).reshape(-1)

    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError("Scores and labels must be non-empty and have equal length.")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and between 0 and 1.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Correctness labels must contain only 0 or 1.")

    return {
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "ece": expected_calibration_error(scores, labels),
        "mean_confidence": float(np.mean(scores)),
        "observed_accuracy": float(np.mean(labels)),
        "sample_count": int(len(scores)),
    }
