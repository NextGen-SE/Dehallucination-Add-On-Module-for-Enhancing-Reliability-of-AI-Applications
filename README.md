# TruthLayer — Claim Extraction (Issues #1–#3) (Person-1)

This folder implements the initial scope of:
- Issue #1: Atomic candidate claim extraction
- Issue #2: Claim-to-source-sentence mapping
- Issue #3: Unit tests

## Important limitation
This is a lightweight, dependency-free baseline using Python's standard library. It uses sentence splitting and conservative heuristics to identify candidate factual statements. It is not a full NLP claim detector, cannot guarantee true factuality or perfect atomicity, and may miss claims or misclassify some statements. It is intended as a working starting point that the team can improve and integrate.

## Files
- `claim_extraction.py`: extraction interface and structured results
- `tests/test_claim_extraction.py`: unit tests
- `examples/demo.py`: runnable example
- `requirements.txt`: optional pytest dependency
- `pyproject.toml`: pytest configuration

## Run the demo
```bash
python examples/demo.py
```

## Run tests without installing packages
```bash
python -m unittest discover -s tests -v
```

Optional pytest:
```bash
python -m pip install pytest
python -m pytest
```

## Example API
```python
from claim_extraction import extract_claims

result = extract_claims("The Earth orbits the Sun. Water contains hydrogen and oxygen.")
print(result.to_dict())
```

Each claim includes `claim_id`, `text`, `source_sentence`, `start_char`, `end_char`, `source_sentence_start`, and `source_sentence_end`.

## Requirement mapping
- REQ-1: accepts up to 2,000 words; raises `InputTooLongError` above the limit.
- REQ-2: extracts candidate factual statements and filters some obvious opinion/rhetorical forms with heuristics.
- REQ-3: preserves the source sentence and character offsets.
- REQ-4: returns `status="no_verifiable_claims"` and the message `No verifiable claims found` if none are identified.

Confirm exact status/error conventions with the API owner before integration.

## Evidence Retrieval and NLI Verification (Person 2)

Takes the `Claim` objects produced by `claim_extraction.py` and decides,
for each one, whether it is **Supported**, **Contradicted**, or
**Unverifiable** against a small knowledge base.

### How it works
1. **Retrieve** (REQ-5): embed the claim and every knowledge-base passage
   with `sentence-transformers/all-MiniLM-L6-v2`, rank passages by cosine
   similarity, keep the top-k (default 5).
2. **Threshold check** (REQ-8): if even the best-ranked passage falls below
   `relevance_threshold` (default 0.35), the claim is marked
   **Unverifiable** immediately — the NLI model is never called, since
   there's nothing relevant to check it against.
3. **Classify** (REQ-6, REQ-7): otherwise, the single best passage and the
   claim are passed to an NLI model
   (`cross-encoder/nli-deberta-v3-small`, via a HuggingFace
   `text-classification` pipeline) which scores `entailment` /
   `contradiction` / `neutral`. The highest-scoring label becomes the
   verdict, and its score becomes the confidence (0.0–1.0).
4. **Evidence** (REQ-10): the supporting/contradicting passage's exact
   text and `source_id` are attached to the result. A `neutral` outcome
   returns **Unverifiable** with no evidence — the model found something
   topically relevant but inconclusive.

### Important limitation
This is a **single-best-passage** baseline: it checks the claim against
only the top-ranked passage, not all `top_k` retrieved ones. That's a
known simplification for this sprint — multi-passage aggregation (e.g.
taking the best entailment/contradiction score across all retrieved
passages, not just the first) is a natural follow-up once we start tuning
against the REQ-19 accuracy target. The relevance threshold (0.35) is also
a starting guess, not an empirically tuned value yet.

### Files
- `evidence_verification.py`: `EvidenceVerifier`, `EvidencePassage`,
  `VerificationResult`, `load_knowledge_base`
- `knowledge_base.json`: small starter knowledge base (swap via REQ-9 by
  pointing `load_knowledge_base` at a different file with the same shape)
- `tests/test_evidence_verification.py`: unit tests against fake
  embedder/classifier (fast, no network), plus one real-model integration
  test that auto-skips if HuggingFace Hub isn't reachable
- `examples/demo_verification.py`: runs Person 1's extractor into Person
  2's verifier end-to-end

### Run the demo
```bash
pip install -r requirements.txt
python examples/demo_verification.py
```

### Run tests
```bash
python -m unittest discover -s tests -v
```
The real-model integration test only runs if `sentence-transformers` /
`transformers` are installed **and** HuggingFace Hub is reachable; it's
skipped automatically otherwise so the rest of the suite never blocks on
network access.

### Requirement mapping
- REQ-5: top-k semantic retrieval over the knowledge base.
- REQ-6: Supported / Contradicted / Unverifiable classification via NLI.
- REQ-7: numeric confidence (0.0–1.0) attached to every verdict.
- REQ-8: Unverifiable when no retrieved passage clears the relevance threshold.
- REQ-9: knowledge base is swappable — just point at a different JSON file.
- REQ-10: exact evidence passage text + source_id returned for Supported/Contradicted.
