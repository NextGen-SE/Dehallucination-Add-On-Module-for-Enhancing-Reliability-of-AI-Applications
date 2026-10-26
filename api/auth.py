"""X-API-Key authentication (REQ-13, NFR-6).

Only SHA-256 digests of keys are configured; plaintext keys are never stored.
Keys are random 32-character alphanumeric strings (SRS Appendix B), so an
unsalted fast hash is adequate here — it would NOT be adequate for
human-chosen passwords.

Generate a new key and its hash:
    python -m api.auth
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import string

API_KEY_HEADER = "X-API-Key"
API_KEY_LENGTH = 32
_ALPHABET = string.ascii_letters + string.digits


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_valid_api_key(key: str | None, allowed_hashes: tuple[str, ...]) -> bool:
    """Constant-time check of `key` against every configured hash."""
    if not key:
        return False
    candidate = hash_api_key(key)
    matched = False
    for allowed in allowed_hashes:
        # No early exit: always compare against every hash.
        matched |= hmac.compare_digest(candidate, allowed)
    return matched


def generate_api_key() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(API_KEY_LENGTH))


if __name__ == "__main__":
    key = generate_api_key()
    print("API key (give to the client; store nowhere else):")
    print(f"  {key}")
    print("SHA-256 hash (put in TRUTHLAYER_API_KEY_HASHES):")
    print(f"  {hash_api_key(key)}")
