# TruthLayer Verification Evaluation (Issue #21)

## Dataset
- File: `data/eval_claims.csv` - 150 labeled claims (50 Supported, 50 Contradicted, 50 Unverifiable).
- Supported/Contradicted claims are paraphrases or falsified versions of facts in `knowledge_base.json`
  (first 75 rows reused from `data/confidence_evaluation.csv`, plus 75 new rows).
- Unverifiable claims are about topics absent from the knowledge base. Labels are relative to the
  knowledge base, not real-world truth: a true claim that is not in the KB (e.g. "Tokyo is the capital of Japan")
  is labeled Unverifiable.
- Labeled by: Shreyas (first 75 rows from the calibration set); reviewed by: <name, or "not independently reviewed">.

## Methodology
- Module under test: `EvidenceVerifier.verify_claim` with real models
  (`all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-small`), top_k=5, relevance threshold=0.35, 15 KB passages.
- Command: `python scripts/evaluate_verification.py`
- Metrics: per-class precision, recall, F1 (scikit-learn), macro average, accuracy, confusion matrix.

## Results
| Verdict | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 1.000 | 1.000 | 1.000 | 50 |
| Contradicted | 0.877 | 1.000 | 0.935 | 50 |
| Unverifiable | 1.000 | 0.860 | 0.925 | 50 |
| **Macro average** | 0.959 | 0.953 | 0.953 | 150 |

Overall accuracy: **0.953** (REQ-19 target: 0.80, met).
Unverifiable claims predicted Supported (NFR-4, must be 0): **0**.

Confusion matrix (rows = expected, columns = predicted):

| expected \ predicted | Supported | Contradicted | Unverifiable |
|---|---|---|---|
| Supported | 50 | 0 | 0 |
| Contradicted | 0 | 50 | 0 |
| Unverifiable | 0 | 7 | 43 |

Total run time: 7.79 s for 150 claims (about 52 ms per claim).

## Analysis
- REQ-19 (accuracy >= 80%) is met at 95.3%. NFR-4 is met: no Unverifiable claim was labeled Supported.
- All 7 errors are one type: an Unverifiable claim predicted Contradicted, with confidence 0.989-1.000.
  Retrieval returns a topically similar passage that passes the 0.35 relevance threshold, and the NLI model
  treats a different fact about a similar subject as a contradiction. Examples: "Jupiter is the largest planet"
  matched the Pacific Ocean passage (KB-013); "Octopuses have three hearts" matched the human heart passage (KB-007);
  "Mumbai is the financial capital of India" matched the Karnataka capital passage (KB-003).
- Contradicted precision (0.877) and Unverifiable recall (0.860) are lower for this reason; Supported is unaffected.
- One of the 7 ("Photosynthesis occurs only during nighttime") is arguably mislabeled: KB-008 states
  photosynthesis uses sunlight, so Contradicted is defensible. The dataset was not changed after seeing the results.
- Possible improvements (not applied, to avoid tuning on the evaluation set): raise the relevance threshold,
  or require the NLI contradiction margin and a subject/entity overlap check before returning Contradicted.

## Limitations
- The knowledge base has only 15 passages; results do not generalise to a large corpus.
- Supported and Contradicted claims are close paraphrases or simple falsifications of KB facts, so the 100%
  scores on those classes are optimistic.
- 150 claims is the SRS minimum, so per-class estimates are uncertain.
- Claims were written by the team, not sampled from HaluEval or TruthfulQA.
- Only the single best passage is checked (see README).
- The dataset overlaps with the calibration data (`data/confidence_evaluation.csv`).