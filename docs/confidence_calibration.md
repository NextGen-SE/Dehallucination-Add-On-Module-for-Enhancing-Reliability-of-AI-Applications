# TruthLayer – Confidence Calibration (Issue #14)

## Objective

Estimate how likely a Supported/Contradicted verdict is to be correct, based on
the verifier's raw confidence. Calibration changes the confidence estimate; it
does not change the predicted verdict.

## Method

This implementation uses **Platt scaling**, implemented as logistic regression
with the raw confidence as its single input feature. The target is:

- `1`: predicted Supported/Contradicted verdict matches the expected verdict.
- `0`: predicted Supported/Contradicted verdict does not match the expected verdict.

The model is fitted only on the calibration subset. The held-out subset is not
used during fitting.

## Why Unverifiable is excluded

The existing verifier can assign Unverifiable confidence through two different
paths: an NLI neutral probability, or `1 - retrieval relevance` when evidence
does not pass the relevance threshold. These values do not represent the same
score source as the NLI confidence for Supported/Contradicted. This first
implementation therefore excludes examples where either the expected or
predicted verdict is Unverifiable. The excluded count is reported.

## Data split and metrics

The evaluation script uses `GroupShuffleSplit` grouped by `source_id`, aiming
to keep claims tied to the same knowledge-base passage out of both subsets.
It reports held-out Brier score and expected calibration error (ECE), along
with mean confidence and observed accuracy. Lower Brier score and ECE are
generally preferable.

## Run

From the repository root:

```powershell
python -m pip install -r requirements-calibration.txt
python scripts/evaluate_calibration.py
python -m unittest tests.test_confidence_calibration -v
```

The real verifier adapters may download Sentence Transformers and Hugging Face
NLI models on first use, so the evaluation script requires those project
dependencies and may require internet access. Unit tests use synthetic scores
and do not download models.

## Output

The evaluation script creates:

- `results/calibration_results.json` – raw predictions, calibrated scores,
  split details, metrics, model parameters, and limitations.
- `results/calibration_report.txt` – readable summary of held-out results.

Generated results are not committed in this ZIP because they must come from
running the model in the target environment.

## Limitations

- The current evaluation dataset contains only 49 examples and requires human
  review for label correctness and evidence alignment.
- The held-out sample may be very small; metrics can vary substantially with
  a small number of examples.
- Source grouping reduces leakage between claims grounded in the same passage,
  but does not remove all semantic similarity or dataset bias.
- This calibrator estimates correctness only for Supported/Contradicted
  predictions. Unverifiable calibration is out of scope for this version.
- Results are exploratory and must not be described as production validation.
