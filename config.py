import os

"""
Lightweight configuration defaults.

The original codebase imported `API_BASE` and `API_KEY` from this module, but
they were not defined. We keep these as optional overrides while still
supporting standard environment variables in `Raccoon/utils.py`.

Secondary semantic chunk leakage (embedding cosine vs chunked hidden prompt) is
configured via `Raccoon.semantic_chunk_leakage.SemanticChunkLeakageConfig.from_env()`
and CLI flags on `run_raccoon_gang.py` / `scripts/backfill_semantic_metrics.py`.

Relevant environment variables (all optional):

- RACCOON_ENABLE_SEMANTIC_CHUNK_LEAKAGE: if "1"/"true", enable semantic metric when
  `run_raccoon_gang.py` runs (still requires OPENAI_API_KEY or
  RACCOON_SEMANTIC_EMBEDDING_API_KEY).
- RACCOON_SEMANTIC_EMBEDDING_API_KEY: overrides OPENAI_API_KEY for embeddings only.
- RACCOON_SEMANTIC_EMBEDDING_BASE_URL: default https://api.openai.com/v1
- RACCOON_SEMANTIC_EMBEDDING_MODEL: default text-embedding-3-small
- RACCOON_SEMANTIC_EMBEDDING_PROVIDER: cache namespace id (default openai)
- RACCOON_SEMANTIC_CACHE_DIR: default .cache/raccoon_semantic_embeddings
- RACCOON_SEMANTIC_CHUNKING_MODE: auto | paragraph | sentence | sliding
- RACCOON_SEMANTIC_MAX_CHUNK_CHARS, RACCOON_SEMANTIC_WINDOW_SIZE,
  RACCOON_SEMANTIC_WINDOW_STRIDE
- RACCOON_SEMANTIC_SIMILARITY_THRESHOLD: v1 default 0.35, v2 default 0.50
- RACCOON_SEMANTIC_FRACTION_THRESHOLD: fraction of chunks above threshold (default 0.25)
- RACCOON_SEMANTIC_TOPK: for top-k mean similarity (default 3)

semantic_metric_v2 (default when RACCOON_SEMANTIC_METRIC_VERSION=v2):

- RACCOON_SEMANTIC_METRIC_VERSION: v1 | v2 (default v2)
- RACCOON_SEMANTIC_MARGIN_THRESHOLD: true_score minus max negative (default 0.10)
- RACCOON_SEMANTIC_NEGATIVE_SAMPLE_COUNT: non-target prompts per sample (default 5)
- RACCOON_SEMANTIC_FINE_MIN_MERGE_CHARS / RACCOON_SEMANTIC_FINE_MAX_MERGED_CHARS
- RACCOON_SEMANTIC_DIAGNOSTIC_THRESHOLD: diagnostic chunk fraction only (default 0.45)
"""

# For OpenAI-compatible APIs. Use "Default" to let the loader choose.
API_BASE = os.getenv("RACCOON_API_BASE", "Default")

# For OpenAI-compatible APIs. Use "Default" to load from provider-specific env vars.
API_KEY = os.getenv("RACCOON_API_KEY", "Default")
