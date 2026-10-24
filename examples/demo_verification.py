"""End-to-end demo: Person 1's claim extraction -> Person 2's evidence verification.

Requires sentence-transformers and transformers to be installed, and
internet access on first run (to download the two pretrained models named
in the SRS). After the first run, the models are cached locally and no
internet is needed.

Run with:
    python examples/demo_verification.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claim_extraction import extract_claims
from evidence_verification import EvidenceVerifier, load_knowledge_base

KB_PATH = Path(__file__).resolve().parents[1] / "knowledge_base.json"

sample_text = (
    "The Earth orbits the Sun. "
    "The Sun orbits the Earth. "
    "Water contains hydrogen and oxygen. "
    "Bananas are a good source of potassium."
)

print("Extracting claims...")
extraction = extract_claims(sample_text)
print(f"Status: {extraction.status}, {len(extraction.claims)} claim(s) found\n")

print("Loading knowledge base and models (first run downloads the models)...")
kb = load_knowledge_base(KB_PATH)
verifier = EvidenceVerifier(kb)

print("Verifying claims...\n")
for result in verifier.verify_claims(extraction.claims):
    print(f"Claim #{result.claim_id}: {result.claim_text}")
    print(f"  Verdict: {result.verdict}  (confidence: {result.confidence})")
    if result.evidence:
        ev = result.evidence[0]
        print(f"  Evidence [{ev.source_id}]: {ev.text}  (relevance: {ev.relevance_score})")
    else:
        print("  Evidence: none")
    print()
