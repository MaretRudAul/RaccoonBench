"""
OpenAI-compatible embedding provider with disk caching (deterministic, batchable).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import List, Protocol, Sequence

import numpy as np
from openai import OpenAI

from Raccoon.text_normalize import normalize_whitespace

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()

# OpenAI (and some OpenAI-compatible APIs) reject "" and may reject whitespace-only
# after server-side strip. Use a deterministic non-empty placeholder for those cases.
_EMBED_EMPTY_PLACEHOLDER = "[RACCOON_EMBED_EMPTY]"


def _embedding_safe_string(s: str) -> str:
    if s is None:
        return _EMBED_EMPTY_PLACEHOLDER
    if not str(s).strip():
        return _EMBED_EMPTY_PLACEHOLDER
    return s


def _cache_key_payload(
    text: str,
    *,
    model: str,
    provider: str,
    normalization: str,
) -> str:
    payload = {
        "v": 2,
        "model": model,
        "provider": provider,
        "normalization": normalization,
        "text": text,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingProvider(Protocol):
    """Protocol for `embed_texts`."""

    def embed_texts(self, texts: List[str]) -> np.ndarray: ...


class CachedOpenAIEmbeddingProvider:
    """
    Embeds text via OpenAI-compatible `client.embeddings.create`.
    Caches each normalized text embedding under cache_dir.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        provider_id: str = "openai",
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
        normalization: str = "ws_collapse",
    ) -> None:
        self.client = client
        self.model = model
        self.provider_id = provider_id
        self.batch_size = max(1, batch_size)
        self.normalization = normalization
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_lock = threading.Lock()

    def _normalize(self, t: str) -> str:
        if self.normalization == "ws_collapse":
            return normalize_whitespace(t)
        return t if t else ""

    def _load_cached(self, key: str) -> np.ndarray | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vec = data.get("embedding")
            if not isinstance(vec, list):
                return None
            return np.asarray(vec, dtype=np.float64)
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", path, e)
            return None

    def _save_cached(self, key: str, vec: np.ndarray) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        try:
            with _CACHE_LOCK:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"embedding": vec.astype(float).tolist(), "model": self.model},
                        f,
                    )
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", path, e)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float64)

        normalized = [_embedding_safe_string(self._normalize(t)) for t in texts]
        keys = [
            _cache_key_payload(
                nt,
                model=self.model,
                provider=self.provider_id,
                normalization=self.normalization,
            )
            for nt in normalized
        ]

        dim: int | None = None
        out_list: List[np.ndarray | None] = [None] * len(normalized)
        to_fetch_idx: List[int] = []
        to_fetch_texts: List[str] = []

        for i, k in enumerate(keys):
            cached = self._load_cached(k)
            if cached is not None:
                out_list[i] = cached
                dim = cached.shape[0]
            else:
                to_fetch_idx.append(i)
                to_fetch_texts.append(normalized[i])

        if to_fetch_texts:
            for start in range(0, len(to_fetch_texts), self.batch_size):
                batch = to_fetch_texts[start : start + self.batch_size]
                batch_idx = to_fetch_idx[start : start + self.batch_size]
                with self._api_lock:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                # Order matches input
                for row, orig_i in zip(resp.data, batch_idx):
                    vec = np.asarray(row.embedding, dtype=np.float64)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    out_list[orig_i] = vec
                    dim = vec.shape[0]
                    self._save_cached(keys[orig_i], vec)

        stacked = np.stack([v for v in out_list if v is not None], axis=0)
        if stacked.shape[0] != len(texts):
            raise RuntimeError("Embedding alignment failure")
        return stacked


def make_semantic_embedding_client(
    *,
    api_key: str,
    base_url: str | None = None,
) -> OpenAI:
    """Build OpenAI client for embeddings (OpenAI API by default)."""
    import httpx

    http_client = httpx.Client(timeout=httpx.Timeout(120.0, read=90.0, write=30.0, connect=10.0))
    return OpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        http_client=http_client,
    )
