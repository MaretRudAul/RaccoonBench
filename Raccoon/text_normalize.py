"""Deterministic text normalization for semantic leakage metrics."""

import re


def normalize_whitespace(text: str) -> str:
    """Strip edges and collapse internal whitespace runs to a single space."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t
