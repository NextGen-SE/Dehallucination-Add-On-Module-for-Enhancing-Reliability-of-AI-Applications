# TruthLayer — Claim Extraction (Issues #1–#3)

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
