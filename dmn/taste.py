"""Taste profile clustering + the dopamine reward function. Tune the constants below freely."""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np

# --- Dopamine reward weights (these are the knobs you'll want to tune) ---
ALPHA = 0.55  # alignment with taste clusters
BETA = 0.25   # novelty (1 - max sim to recent findings)
GAMMA = 0.15  # surprise (taste-aligned but new angle)
EPS = 0.05    # serendipity epsilon-greedy probability + bonus weight

MIN_CLUSTERS = 4
DEFAULT_K = 8


def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, returning the original if its norm is zero."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(_norm(a), _norm(b)))


def alignment(item_emb: np.ndarray, centroids: Sequence[np.ndarray]) -> float:
    """Max cosine similarity between an item embedding and any taste cluster centroid."""
    if centroids is None or len(centroids) == 0:
        return 0.0
    return max(_cos(item_emb, c) for c in centroids)


def novelty(item_emb: np.ndarray, recent: Sequence[np.ndarray]) -> float:
    """1 - max cosine similarity to recent findings (high = distinct from past entries)."""
    if recent is None or len(recent) == 0:
        return 1.0
    return float(1.0 - max(_cos(item_emb, r) for r in recent))


def surprise(
    item_emb: np.ndarray,
    recent: Sequence[np.ndarray],
    centroids: Sequence[np.ndarray],
) -> float:
    """Reward items more aligned with taste than with the user's recent finds (a 'new angle')."""
    if centroids is None or len(centroids) == 0:
        return 0.0
    a = alignment(item_emb, centroids)
    if recent is None or len(recent) == 0:
        return float(a * 0.5)
    near_recent = max(_cos(item_emb, r) for r in recent)
    return float(max(0.0, a - near_recent))


def cluster(
    embeddings: np.ndarray,
    k: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means cluster embeddings; returns (centroids, labels). `sample_weight` (v0.1.3) lets
    high-visit-count interests pull centroids more strongly. Falls back to raw rows for tiny inputs.
    """
    if embeddings is None or len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=int)
    n = len(embeddings)
    if n < MIN_CLUSTERS:
        return embeddings.astype(np.float32).copy(), np.arange(n)
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        return embeddings.astype(np.float32).copy(), np.arange(n)
    k = k or min(DEFAULT_K, n)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw.shape[0] == n and np.any(sw > 0):
            labels = km.fit_predict(embeddings, sample_weight=sw)
        else:
            labels = km.fit_predict(embeddings)
    else:
        labels = km.fit_predict(embeddings)
    return km.cluster_centers_.astype(np.float32), labels


def dopamine(
    item_emb: np.ndarray,
    centroids: Sequence[np.ndarray],
    recent: Sequence[np.ndarray],
    rng: Optional[random.Random] = None,
) -> dict:
    """Compute the dopamine reward as a dict with breakdown + total. Logged with every brief."""
    rng = rng or random.Random()
    a = alignment(item_emb, centroids)
    n = novelty(item_emb, recent)
    s = surprise(item_emb, recent, centroids)
    serendipity = 0.0
    # Epsilon-greedy: small chance to amplify a low-alignment, high-novelty item
    if rng.random() < EPS and a < 0.4 and n > 0.6:
        serendipity = 1.0
    total = ALPHA * a + BETA * n + GAMMA * s + EPS * serendipity
    return {
        "alignment": round(float(a), 4),
        "novelty": round(float(n), 4),
        "surprise": round(float(s), 4),
        "serendipity": round(float(serendipity), 4),
        "total": round(float(total), 4),
    }


def nudge_centroid(
    centroid: np.ndarray,
    brief_emb: np.ndarray,
    step: float = 0.05,
    weight: float = 1.0,
) -> np.ndarray:
    """Move a cluster centroid a small step toward a high-dopamine brief embedding.

    `weight` (v0.1.3) scales the step: a brief tied to a heavy-visit-count interest moves
    the centroid harder. Clamped to [0, 0.5] so a wildly weighted brief can't flip a cluster.
    """
    effective_step = max(0.0, min(0.5, float(step) * float(weight)))
    new = (1.0 - effective_step) * np.asarray(centroid, dtype=np.float32) + effective_step * np.asarray(
        brief_emb, dtype=np.float32
    )
    return _norm(new).astype(np.float32)


def best_cluster_index(
    centroids: Sequence[np.ndarray], item_emb: np.ndarray
) -> Optional[int]:
    """Return the index of the cluster centroid with maximum cosine similarity to the item."""
    if centroids is None or len(centroids) == 0:
        return None
    return max(range(len(centroids)), key=lambda i: _cos(centroids[i], item_emb))
