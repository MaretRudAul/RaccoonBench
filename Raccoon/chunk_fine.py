"""
Fine-grained deterministic chunking for semantic_metric_v2.

Paragraph → sentences → merge into windows bounded by min/max character targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from Raccoon.text_normalize import normalize_whitespace


@dataclass(frozen=True)
class TextChunk:
    """One chunk with inspectable provenance."""

    text: str
    index: int
    paragraph_index: int
    sentence_start: int  # global flat sentence index (start inclusive)
    sentence_end: int  # global flat sentence index (end exclusive)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for s in raw:
        n = normalize_whitespace(s)
        if n:
            out.append(n)
    if not out and text.strip():
        return [normalize_whitespace(text)]
    return out


def _paragraphs(text: str) -> List[str]:
    raw_paras = re.split(r"\n\s*\n+", text.strip())
    return [normalize_whitespace(p) for p in raw_paras if normalize_whitespace(p)]


def _chunks_for_paragraph(
    sentences: List[str],
    *,
    paragraph_index: int,
    flat_sentence_base: int,
    min_merge_chars: int,
    max_merged_chars: int,
    global_chunk_index: int,
) -> tuple[List[TextChunk], int]:
    """
    Greedily pack sentences: each chunk grows until >= min_merge_chars or next
    sentence would exceed max_merged_chars; then emit and continue.
    """
    chunks: List[TextChunk] = []
    i = 0
    n = len(sentences)
    gidx = global_chunk_index
    while i < n:
        start_i = i
        buf: List[str] = []
        acc = 0
        while i < n:
            s = sentences[i].strip()
            if not s:
                i += 1
                continue
            add = len(s) + (1 if buf else 0)
            if buf and acc + add > max_merged_chars:
                break
            buf.append(s)
            acc += add
            i += 1
            if acc >= min_merge_chars:
                break
        if buf:
            t = normalize_whitespace(" ".join(buf))
            if t:
                chunks.append(
                    TextChunk(
                        text=t,
                        index=gidx,
                        paragraph_index=paragraph_index,
                        sentence_start=flat_sentence_base + start_i,
                        sentence_end=flat_sentence_base + i,
                    )
                )
                gidx += 1
        elif i < n:
            i += 1
    return chunks, gidx


def chunk_hidden_prompt_fine(
    text: str,
    *,
    min_merge_chars: int = 40,
    max_merged_chars: int = 320,
) -> List[TextChunk]:
    """Sentence-based chunks with bounded merging; respects paragraph breaks."""
    if not (text or "").strip():
        return []

    paragraphs = _paragraphs(text)
    if not paragraphs:
        return []

    all_chunks: List[TextChunk] = []
    flat_sent = 0
    gidx = 0
    for pi, para in enumerate(paragraphs):
        sents = _split_sentences(para)
        if not sents:
            continue
        new_chunks, gidx = _chunks_for_paragraph(
            sents,
            paragraph_index=pi,
            flat_sentence_base=flat_sent,
            min_merge_chars=min_merge_chars,
            max_merged_chars=max_merged_chars,
            global_chunk_index=gidx,
        )
        all_chunks.extend(new_chunks)
        flat_sent += len(sents)

    # Fix sequential indices
    return [
        TextChunk(
            text=c.text,
            index=ii,
            paragraph_index=c.paragraph_index,
            sentence_start=c.sentence_start,
            sentence_end=c.sentence_end,
        )
        for ii, c in enumerate(all_chunks)
        if c.text.strip()
    ]


def chunk_response_text(
    text: str,
    *,
    min_merge_chars: int = 40,
    max_merged_chars: int = 320,
) -> List[TextChunk]:
    """Same strategy as hidden prompt (paragraphs if any, else whole text)."""
    return chunk_hidden_prompt_fine(
        text,
        min_merge_chars=min_merge_chars,
        max_merged_chars=max_merged_chars,
    )
