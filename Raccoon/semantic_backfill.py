"""
Recompute semantic_chunk_leakage on saved benchmark JSON (no new victim-model calls).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from Raccoon.semantic_chunk_leakage import (
    SemanticChunkLeakageConfig,
    compute_chunk_semantic_scores,
)
from Raccoon.semantic_embedding import CachedOpenAIEmbeddingProvider, EmbeddingProvider

logger = logging.getLogger(__name__)


def enrich_attack_info_attempt(
    attempt: Dict[str, Any],
    embedder: EmbeddingProvider,
    config: SemanticChunkLeakageConfig,
    *,
    force: bool = False,
) -> bool:
    """
    Mutates attempt with semantic_chunk_leakage if needed.

    Returns True if a new computation was run.
    """
    if not force:
        existing = attempt.get("semantic_chunk_leakage")
        if isinstance(existing, dict) and existing.get("error") is None:
            if existing.get("max_chunk_similarity") is not None or existing.get(
                "num_chunks"
            ) is not None:
                return False

    hidden = attempt.get("prompt") or ""
    resp = attempt.get("parsed_response")
    if resp is None:
        resp = attempt.get("response") or ""
    attempt["semantic_chunk_leakage"] = compute_chunk_semantic_scores(
        hidden, resp, embedder, config
    )
    return True


def enrich_result_file(
    path: Path,
    embedder: EmbeddingProvider,
    config: SemanticChunkLeakageConfig,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    n = 0
    for run in payload.get("runs", []):
        for attempt in run.get("atk_info", []):
            if enrich_attack_info_attempt(attempt, embedder, config, force=force):
                n += 1

    logger.info("Updated %d attempts in %s", n, path.name)
    return {"path": str(path), "attempts_updated": n, "payload": payload}


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def run_backfill(
    results_dir: Path,
    embedder: CachedOpenAIEmbeddingProvider,
    config: SemanticChunkLeakageConfig,
    *,
    output_dir: Path | None = None,
    in_place: bool = False,
    force: bool = False,
) -> None:
    files = sorted(results_dir.glob("atk_*_def_*.json"))
    if not files:
        raise FileNotFoundError(f"No atk_*_def_*.json under {results_dir}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        info = enrich_result_file(fp, embedder, config, force=force)
        payload = info["payload"]
        if in_place:
            write_payload(fp, payload)
        elif output_dir is not None:
            write_payload(output_dir / fp.name, payload)
        else:
            raise ValueError("Specify --in_place or --output_dir")
