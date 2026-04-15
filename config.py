import os

"""
Lightweight configuration defaults.

The original codebase imported `API_BASE` and `API_KEY` from this module, but
they were not defined. We keep these as optional overrides while still
supporting standard environment variables in `Raccoon/utils.py`.

Secondary semantic leakage (semantic_metric_v2: pairwise chunk cosine vs hidden
prompt, with negative controls) is configured via
`Raccoon.semantic_chunk_leakage.SemanticChunkLeakageConfig.from_env()` and CLI
flags on `run_raccoon_gang.py` / `scripts/backfill_semantic_metrics.py`.

Relevant environment variables (all optional):

- RACCOON_ENABLE_SEMANTIC_CHUNK_LEAKAGE: if "1"/"true", enable when
  `run_raccoon_gang.py` runs (requires OPENAI_API_KEY or
  RACCOON_SEMANTIC_EMBEDDING_API_KEY).
- RACCOON_SEMANTIC_EMBEDDING_API_KEY: overrides OPENAI_API_KEY for embeddings only.
- RACCOON_SEMANTIC_EMBEDDING_BASE_URL: default https://api.openai.com/v1
- RACCOON_SEMANTIC_EMBEDDING_MODEL: default text-embedding-3-small
- RACCOON_SEMANTIC_EMBEDDING_PROVIDER: cache namespace id (default openai)
- RACCOON_SEMANTIC_CACHE_DIR: default .cache/raccoon_semantic_embeddings
- RACCOON_SEMANTIC_SIMILARITY_THRESHOLD: minimum true score for candidate (default 0.50)
- RACCOON_SEMANTIC_MARGIN_THRESHOLD: true_score minus max negative (default 0.10)
- RACCOON_SEMANTIC_NEGATIVE_SAMPLE_COUNT: non-target prompts per sample (default 5)
- RACCOON_SEMANTIC_TOPK: top-k mean over flattened pairwise matrix (default 3)
- RACCOON_SEMANTIC_FINE_MIN_MERGE_CHARS / RACCOON_SEMANTIC_FINE_MAX_MERGED_CHARS
- RACCOON_SEMANTIC_DIAGNOSTIC_THRESHOLD: diagnostic chunk fraction only (default 0.45)
"""

# For OpenAI-compatible APIs. Use "Default" to let the loader choose.
API_BASE = os.getenv("RACCOON_API_BASE", "Default")

# For OpenAI-compatible APIs. Use "Default" to load from provider-specific env vars.
API_KEY = os.getenv("RACCOON_API_KEY", "Default")
