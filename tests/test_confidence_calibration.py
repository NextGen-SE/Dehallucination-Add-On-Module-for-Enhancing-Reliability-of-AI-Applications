"""Unit tests for confidence calibration (no model downloads required)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import ConfidenceCalibrator, calibration_metrics, expected_calibration_error


class TestConfidenceCalibrator(unittest.TestCase):
    def setUp(self):
        self.raw = [0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95]
        self.correct = [0, 0, 0, 1, 0, 1, 1, 1]

    def test_fit_and_predict_returns_probabilities(self):
        calibrator = ConfidenceCalibrator().fit(self.raw, self.correct)
        calibrated = calibrator.predict_calibrated([0.60, 0.80, 0.95])
        self.assertEqual(len(calibrated), 3)
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertTrue(np.all(calibrated <= 1.0))

    def test_parameters_are_available_after_fit(self):
        calibrator = ConfidenceCalibrator().fit(self.raw, self.correct)
        self.assertEqual(
            calibrator.parameters["method"],
            "Platt scaling (logistic regression)",
        )

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            ConfidenceCalibrator().predict_calibrated([0.7])

    def test_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValueError):
            ConfidenceCalibrator().fit([0.2, 0.4, 1.2, 0.8], [0, 0, 1, 1])

    def test_rejects_non_binary_labels(self):
        with self.assertRaises(ValueError):
            ConfidenceCalibrator().fit([0.2, 0.4, 0.6, 0.8], [0, 1, 2, 1])

    def test_requires_both_correctness_classes(self):
        with self.assertRaises(ValueError):
            ConfidenceCalibrator().fit([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1])

    def test_metrics_are_computed(self):
        metrics = calibration_metrics([0.8, 0.4], [1, 0])
        self.assertAlmostEqual(metrics["brier_score"], 0.10)
        self.assertEqual(metrics["sample_count"], 2)
        self.assertIn("ece", metrics)

    def test_ece_rejects_invalid_bin_count(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([0.5], [1], n_bins=0)


if __name__ == "__main__":
    unittest.main()
