from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claim_extraction import extract_claims

sample = (
    "The Earth orbits the Sun. "
    "Water contains hydrogen and oxygen. "
    "I think this is wonderful. "
    "Is the Moon made of cheese?"
)
result = extract_claims(sample)
print(f"Status: {result.status}")
if result.message:
    print(f"Message: {result.message}")
for claim in result.claims:
    print(f"\nClaim #{claim.claim_id}: {claim.text}")
    print(f"Source sentence: {claim.source_sentence}")
    print(f"Offsets: [{claim.start_char}, {claim.end_char})")
