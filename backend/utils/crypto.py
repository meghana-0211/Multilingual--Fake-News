"""
backend/utils/crypto.py

Cryptographic utilities.
Centralises the SHA-256 hashing logic so routes, blockchain bridge,
and tests all use the identical normalisation algorithm.
"""

import hashlib


def hash_text(text: str) -> str:
    """
    Return the SHA-256 hex digest of `text`.

    Normalisation (must stay in sync with blockchain/web3_client.py):
        - strip leading/trailing whitespace
        - collapse internal whitespace to single spaces
        - encode as UTF-8

    Returns
    -------
    str  – 64-character lowercase hex string
    """
    normalised = " ".join(text.strip().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def hash_text_bytes(text: str) -> bytes:
    """Same as hash_text() but returns raw bytes (for blockchain calls)."""
    return bytes.fromhex(hash_text(text))