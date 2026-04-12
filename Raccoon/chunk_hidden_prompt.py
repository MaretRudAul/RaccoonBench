"""
Deterministic chunking of hidden English prompts for chunk-based semantic leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from Raccoon.text_normalize import normalize_whitespace


@dataclass(frozen=True)
class ChunkMetadata:
    """Per-chunk index and optional source hint for inspection."""

    index: int
    source: str  # "paragraph" | "sentence" | "window"


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Split on sentence boundaries; keep order, drop empties
    raw = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for s in raw:
        n = normalize_whitespace(s)
        if n:
            out.append(n)
    if not out and text.strip():
        return [normalize_whitespace(text)]
    return out


def _sliding_windows(text: str, window_size: int, stride: int) -> List[str]:
    t = normalize_whitespace(text)
    if not t:
        return []
    if len(t) <= window_size:
        return [t]
    windows: List[str] = []
    i = 0
    while i < len(t):
        chunk = t[i : i + window_size]
        if normalize_whitespace(chunk):
            windows.append(chunk.strip())
        if i + window_size >= len(t):
            break
        i += stride
    return windows


def chunk_hidden_prompt(
    text: str,
    *,
    mode: str = "auto",
    max_chunk_chars: int = 1200,
    window_size: int = 400,
    window_stride: int = 200,
) -> Tuple[List[str], List[ChunkMetadata]]:
    """
    Split hidden prompt into ordered, non-empty chunks (deterministic).

    Strategy when mode == "auto":
    1. Split on paragraph breaks (blank lines) before global whitespace collapse.
    2. Normalize whitespace within each piece.
    3. Any paragraph longer than max_chunk_chars -> sentence split.
    4. Any sentence still longer than max_chunk_chars -> sliding windows.

    Returns:
        chunks: list of chunk strings
        meta: parallel ChunkMetadata (index, source)
    """
    if not (text or "").strip():
        return [], []

    if mode not in ("auto", "paragraph", "sentence", "sliding"):
        raise ValueError(f"Unknown chunking mode: {mode}")

    if mode == "sliding":
        normalized = normalize_whitespace(text)
        if not normalized:
            return [], []
        ws = _sliding_windows(normalized, window_size, window_stride)
        return ws, [ChunkMetadata(i, "window") for i in range(len(ws))]

    # Paragraphs: split raw text on blank lines, then normalize each paragraph.
    raw_paras = re.split(r"\n\s*\n+", text.strip())
    paragraphs = [normalize_whitespace(p) for p in raw_paras if normalize_whitespace(p)]
    if not paragraphs:
        return [], []

    if mode == "paragraph":
        return paragraphs, [ChunkMetadata(i, "paragraph") for i in range(len(paragraphs))]

    if mode == "sentence":
        blob = " ".join(paragraphs)
        final_chunks: List[str] = []
        final_meta: List[ChunkMetadata] = []
        idx = 0
        for s in _split_sentences(blob):
            if len(s) <= max_chunk_chars:
                final_chunks.append(s)
                final_meta.append(ChunkMetadata(idx, "sentence"))
                idx += 1
            else:
                for w in _sliding_windows(s, window_size, window_stride):
                    final_chunks.append(w)
                    final_meta.append(ChunkMetadata(idx, "window"))
                    idx += 1
        return final_chunks, final_meta

    # auto: start from paragraphs
    pieces: List[Tuple[str, str]] = []
    for p in paragraphs:
        if len(p) <= max_chunk_chars:
            pieces.append((p, "paragraph"))
        else:
            for s in _split_sentences(p):
                pieces.append((s, "sentence"))

    final_chunks: List[str] = []
    final_meta: List[ChunkMetadata] = []
    idx = 0
    for content, src in pieces:
        if len(content) <= max_chunk_chars:
            final_chunks.append(content)
            final_meta.append(ChunkMetadata(idx, src))
            idx += 1
        else:
            for w in _sliding_windows(content, window_size, window_stride):
                final_chunks.append(w)
                final_meta.append(ChunkMetadata(idx, "window"))
                idx += 1

    return final_chunks, final_meta
