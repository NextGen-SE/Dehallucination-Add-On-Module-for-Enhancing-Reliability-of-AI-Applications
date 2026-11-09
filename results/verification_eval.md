| Verdict | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 1.000 | 1.000 | 1.000 | 50 |
| Contradicted | 0.877 | 1.000 | 0.935 | 50 |
| Unverifiable | 1.000 | 0.860 | 0.925 | 50 |
| **Macro average** | 0.959 | 0.953 | 0.953 | 150 |

Overall accuracy: **0.953** (REQ-19 target: 0.80, met)
Unverifiable claims predicted Supported (NFR-4, must be 0): **0**

Confusion matrix (rows = expected, columns = predicted):

| expected \ predicted | Supported | Contradicted | Unverifiable |
|---|---|---|---|
| Supported | 50 | 0 | 0 |
| Contradicted | 0 | 50 | 0 |
| Unverifiable | 0 | 7 | 43 |

Config: top_k=5, relevance_threshold=0.35, knowledge base passages=15
