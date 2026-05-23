"""Embedding backends. Defaults to sentence-transformers (local), with OpenAI + hash fallbacks."""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np

DEFAULT_DIM = 384  # all-MiniLM-L6-v2

_MODEL_CACHE: dict = {}


def _get_st_model():
    """Return a cached sentence-transformers model, or None if package missing."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    if "st" not in _MODEL_CACHE:
        _MODEL_CACHE["st"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_CACHE["st"]


def _hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic fallback embedding so dry-run works with no ML dependencies."""
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def embed(texts: Sequence[str]) -> np.ndarray:
    """Embed a list of texts; returns an (n, d) float32 numpy array (rows are L2-normalized)."""
    backend = os.environ.get("DMN_EMBEDDINGS", "st").lower()
    texts = [t or "" for t in texts]
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "hash":
        return np.stack([_hash_embed(t) for t in texts])
    model = _get_st_model()
    if model is None:
        return np.stack([_hash_embed(t) for t in texts])
    arr = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    return arr


def _embed_openai(texts: Sequence[str]) -> np.ndarray:
    """Embed via OpenAI text-embedding-3-small. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("openai not installed; `uv add openai`") from e
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=list(texts))
    out = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms
