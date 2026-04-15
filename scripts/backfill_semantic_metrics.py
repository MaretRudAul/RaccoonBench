#!/usr/bin/env python3
"""
Backfill semantic_chunk_leakage_v2 (pairwise chunk similarity + negative controls)
onto saved RaccoonBench JSON results.

Does not rerun victim models. Requires embedding API key.

  python scripts/backfill_semantic_metrics.py \\
    --results_dir results/run_YYYYMMDD_HHMMSS --in_place
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    pass

from Raccoon.semantic_chunk_leakage import SemanticChunkLeakageConfig
from Raccoon.semantic_embedding import (
    CachedOpenAIEmbeddingProvider,
    make_semantic_embedding_client,
)
from Raccoon.semantic_metric_v2 import compute_semantic_metric_v2


def _pool_from_payload(payload: dict) -> list[str]:
    seen: list[str] = []
    for r in payload.get("runs", []):
        for a in r.get("atk_info", []):
            p = a.get("prompt")
            if isinstance(p, str) and p.strip():
                seen.append(p)
    out: list[str] = []
    u: set[str] = set()
    for s in seen:
        if s not in u:
            u.add(s)
            out.append(s)
    return out


def _build_config(args: argparse.Namespace) -> SemanticChunkLeakageConfig:
    c = SemanticChunkLeakageConfig.from_env()
    c.enabled = True
    if args.semantic_similarity_threshold is not None:
        c.semantic_similarity_threshold = args.semantic_similarity_threshold
    if args.semantic_margin_threshold is not None:
        c.semantic_margin_threshold = args.semantic_margin_threshold
    if args.negative_prompt_sample_count is not None:
        c.negative_prompt_sample_count = args.negative_prompt_sample_count
    if args.semantic_topk is not None:
        c.semantic_topk = args.semantic_topk
    if args.fine_min_merge_chars is not None:
        c.fine_min_merge_chars = args.fine_min_merge_chars
    if args.fine_max_merged_chars is not None:
        c.fine_max_merged_chars = args.fine_max_merged_chars
    if args.diagnostic_similarity_threshold is not None:
        c.diagnostic_similarity_threshold = args.diagnostic_similarity_threshold
    if args.semantic_cache_dir is not None:
        c.cache_dir = args.semantic_cache_dir
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=str, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_place", action="store_true")
    g.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--semantic_similarity_threshold", type=float, default=None)
    ap.add_argument("--semantic_margin_threshold", type=float, default=None)
    ap.add_argument("--negative_prompt_sample_count", type=int, default=None)
    ap.add_argument("--semantic_topk", type=int, default=None)
    ap.add_argument("--fine_min_merge_chars", type=int, default=None)
    ap.add_argument("--fine_max_merged_chars", type=int, default=None)
    ap.add_argument("--diagnostic_similarity_threshold", type=float, default=None)
    ap.add_argument("--semantic_cache_dir", type=str, default=None)
    args = ap.parse_args()

    emb_key = os.getenv("RACCOON_SEMANTIC_EMBEDDING_API_KEY") or os.getenv(
        "OPENAI_API_KEY"
    )
    if not emb_key:
        raise SystemExit("Need OPENAI_API_KEY or RACCOON_SEMANTIC_EMBEDDING_API_KEY.")

    config = _build_config(args)
    client = make_semantic_embedding_client(
        api_key=emb_key,
        base_url=os.getenv("RACCOON_SEMANTIC_EMBEDDING_BASE_URL"),
    )
    embedder = CachedOpenAIEmbeddingProvider(
        client,
        model=config.embedding_model,
        provider_id=config.embedding_provider_id,
        cache_dir=config.cache_dir,
    )

    rd = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for fp in sorted(rd.glob("atk_*_def_*.json")):
        with open(fp, encoding="utf-8") as f:
            payload = json.load(f)
        pool = _pool_from_payload(payload)
        for run in payload.get("runs", []):
            gname = run.get("gpts_name", "unknown")
            for att in run.get("atk_info", []):
                if not args.force and att.get("semantic_chunk_leakage_v2"):
                    sc = att["semantic_chunk_leakage_v2"]
                    if isinstance(sc, dict) and sc.get("error") is None and sc.get(
                        "metric_version"
                    ):
                        continue
                hidden = att.get("prompt") or ""
                resp = att.get("parsed_response")
                if resp is None:
                    resp = att.get("response") or ""
                att["semantic_chunk_leakage_v2"] = compute_semantic_metric_v2(
                    hidden,
                    resp,
                    embedder,
                    config,
                    other_hidden_prompts=pool,
                    sample_key=str(gname),
                )
                total += 1
        target = fp if args.in_place else (out_dir / fp.name)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    print(f"Updated {total} attempts (semantic_chunk_leakage_v2).")


if __name__ == "__main__":
    main()
