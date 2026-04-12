#!/usr/bin/env python3
"""
Backfill secondary semantic_chunk_leakage metrics onto saved RaccoonBench JSON results.

Does not rerun victim-model inference; only calls the embedding API for missing vectors
(subject to on-disk cache under RACCOON_SEMANTIC_CACHE_DIR).

Examples:
  python scripts/backfill_semantic_metrics.py \\
    --results_dir results/run_gpt_3.5-turbo_bengali --in_place

  python scripts/backfill_semantic_metrics.py \\
    --results_dir results/run_20260411_133310 \\
    --output_dir results/run_20260411_133310_semantic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    pass

from Raccoon.semantic_backfill import run_backfill
from Raccoon.semantic_chunk_leakage import SemanticChunkLeakageConfig
from Raccoon.semantic_embedding import (
    CachedOpenAIEmbeddingProvider,
    make_semantic_embedding_client,
)


def _build_config(args: argparse.Namespace) -> SemanticChunkLeakageConfig:
    c = SemanticChunkLeakageConfig.from_env()
    c.enabled = True
    if args.semantic_similarity_threshold is not None:
        c.semantic_similarity_threshold = args.semantic_similarity_threshold
    if args.semantic_fraction_threshold is not None:
        c.semantic_fraction_threshold = args.semantic_fraction_threshold
    if args.semantic_topk is not None:
        c.semantic_topk = args.semantic_topk
    if args.semantic_chunking_mode is not None:
        c.chunking_mode = args.semantic_chunking_mode
    if args.semantic_embedding_model is not None:
        c.embedding_model = args.semantic_embedding_model
    if args.semantic_max_chunk_chars is not None:
        c.max_chunk_chars = args.semantic_max_chunk_chars
    if args.semantic_window_size is not None:
        c.window_size = args.semantic_window_size
    if args.semantic_window_stride is not None:
        c.window_stride = args.semantic_window_stride
    if args.semantic_cache_dir is not None:
        c.cache_dir = args.semantic_cache_dir
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing atk_*_def_*.json files",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite JSON files in results_dir",
    )
    g.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write enriched copies here (keeps originals unchanged)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when semantic_chunk_leakage already present",
    )
    ap.add_argument("--semantic_similarity_threshold", type=float, default=None)
    ap.add_argument("--semantic_fraction_threshold", type=float, default=None)
    ap.add_argument("--semantic_topk", type=int, default=None)
    ap.add_argument("--semantic_chunking_mode", type=str, default=None)
    ap.add_argument("--semantic_embedding_model", type=str, default=None)
    ap.add_argument("--semantic_max_chunk_chars", type=int, default=None)
    ap.add_argument("--semantic_window_size", type=int, default=None)
    ap.add_argument("--semantic_window_stride", type=int, default=None)
    ap.add_argument("--semantic_cache_dir", type=str, default=None)
    args = ap.parse_args()

    emb_key = os.getenv("RACCOON_SEMANTIC_EMBEDDING_API_KEY") or os.getenv(
        "OPENAI_API_KEY"
    )
    if not emb_key:
        raise SystemExit(
            "Need OPENAI_API_KEY or RACCOON_SEMANTIC_EMBEDDING_API_KEY for embeddings."
        )

    config = _build_config(args)
    emb_base = os.getenv("RACCOON_SEMANTIC_EMBEDDING_BASE_URL")
    client = make_semantic_embedding_client(api_key=emb_key, base_url=emb_base)
    embedder = CachedOpenAIEmbeddingProvider(
        client,
        model=config.embedding_model,
        provider_id=config.embedding_provider_id,
        cache_dir=config.cache_dir,
    )

    out = Path(args.output_dir) if args.output_dir else None
    run_backfill(
        Path(args.results_dir),
        embedder,
        config,
        output_dir=out,
        in_place=args.in_place,
        force=args.force,
    )
    print("Done.")


if __name__ == "__main__":
    main()
