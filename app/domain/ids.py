"""Stable identifiers for domain values."""

import hashlib


def stable_id(namespace: str, *parts: str) -> str:
    """Return a namespaced ID derived from separator-delimited UTF-8 parts."""
    payload = "\x1f".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{namespace}_{digest}"
