"""Taste profile clustering + the dopamine reward function. Tune the constants below freely."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# --- Dopamine reward weights (these are the knobs you'll want to tune) ---
ALPHA = 0.50  # alignment with taste clusters
BETA = 0.22   # novelty (1 - max sim to recent findings)
GAMMA = 0.13  # surprise (taste-aligned but new angle)
DELTA = 0.20  # fulfillment — did research/tools return useful material?
# Serendipity used to overload a single EPS constant as *both* the firing probability and
# the score weight, which capped its contribution at 0.05 — too small to widen horizons.
# These are now independent knobs: how often it fires vs. how much it's worth when it does.
EPS = 0.05               # P(serendipity fires) when the low-align/high-novelty gate is met
SERENDIPITY_BONUS = 0.10  # weight added to `total` when serendipity fires

# Serendipity bonus only applies when fulfillment meets this floor.
SERENDIPITY_MIN_FULFILLMENT = 0.35

MIN_CLUSTERS = 4
DEFAULT_K = 8


@dataclass
class ClusterResult:
    """Output of `cluster()` — centroids are medoid embeddings for HDBSCAN."""

    centroids: np.ndarray
    labels: np.ndarray
    medoid_indices: list[int]
    method: str
    n_noise: int
    silhouette: float | None


def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, returning the original if its norm is zero."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(_norm(a), _norm(b)))


def recency_decay(days_since: float, half_life_days: float = 90.0) -> float:
    """Exponential recency weight: 0.5 ** (days_since / half_life_days)."""
    return 0.5 ** (float(days_since) / float(half_life_days))


def composite_weight(
    visit_count: float,
    days_since: float,
    source: str,
    half_life_days: float = 90.0,
) -> float:
    """Composite sample weight: log1p(visit_count) × recency factor.

    Manual / interview sources are treated as maximally recent; journal_import gets 0.95.
    """
    base = math.log1p(max(0.0, float(visit_count)))
    if source.startswith("manual") or source == "interview":
        recency = 1.0
    elif source == "journal_import":
        recency = 0.95
    else:
        recency = recency_decay(days_since, half_life_days)
    return base * recency


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


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def _weighted_medoid(
    indices: list[int],
    embeddings: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> int:
    """Return the index (into `embeddings`) of the weighted medoid for a member set."""
    if len(indices) == 1:
        return indices[0]
    sub = embeddings[indices]
    w = (
        np.asarray(sample_weight, dtype=np.float64)[indices]
        if sample_weight is not None
        else np.ones(len(indices), dtype=np.float64)
    )
    w = np.maximum(w, 1e-8)
    # Weighted sum of squared Euclidean distances on normalized vectors.
    sub_n = _normalize_rows(sub.astype(np.float64))
    dists = np.zeros(len(indices), dtype=np.float64)
    for i in range(len(indices)):
        diff = sub_n - sub_n[i]
        dists[i] = float(np.sum(w * np.sum(diff * diff, axis=1)))
    return indices[int(np.argmin(dists))]


def _build_from_labels(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_weight: Optional[np.ndarray],
    method: str,
    include_noise_cluster: bool = False,
) -> ClusterResult:
    """Build centroids (medoids), medoid_indices, and silhouette from hard labels."""
    n = len(embeddings)
    unique = sorted(set(int(l) for l in labels if int(l) >= 0))
    medoid_indices: list[int] = []
    centroids: list[np.ndarray] = []
    out_labels = labels.copy()

    for lab in unique:
        members = [i for i in range(n) if int(labels[i]) == lab]
        midx = _weighted_medoid(members, embeddings, sample_weight)
        medoid_indices.append(midx)
        centroids.append(embeddings[midx].astype(np.float32).copy())

    n_noise = int(np.sum(labels == -1))

    if include_noise_cluster and n_noise > 0:
        noise_members = [i for i in range(n) if int(labels[i]) == -1]
        midx = _weighted_medoid(noise_members, embeddings, sample_weight)
        medoid_indices.append(midx)
        centroids.append(embeddings[midx].astype(np.float32).copy())
        noise_cluster_id = max(unique) + 1 if unique else 0
        for i in noise_members:
            out_labels[i] = noise_cluster_id

    silhouette = _silhouette(embeddings, labels)

    return ClusterResult(
        centroids=np.stack(centroids).astype(np.float32) if centroids else np.zeros((0, embeddings.shape[1]), dtype=np.float32),
        labels=out_labels,
        medoid_indices=medoid_indices,
        method=method,
        n_noise=n_noise,
        silhouette=silhouette,
    )


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette on non-noise points when >= 2 clusters exist."""
    mask = labels >= 0
    if int(mask.sum()) < 2:
        return None
    sub_labels = labels[mask]
    n_clusters = len(set(int(l) for l in sub_labels))
    if n_clusters < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score  # type: ignore

        sub_emb = _normalize_rows(embeddings[mask].astype(np.float64))
        return float(silhouette_score(sub_emb, sub_labels, metric="euclidean"))
    except Exception:
        return None


def _cluster_hdbscan(
    embeddings: np.ndarray,
    sample_weight: Optional[np.ndarray],
    min_cluster_size: Optional[int] = None,
) -> ClusterResult:
    """HDBSCAN on L2-normalized embeddings with sqrt(weight) row scaling."""
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return _cluster_kmeans(embeddings, k=None, sample_weight=sample_weight)

    n = len(embeddings)
    if min_cluster_size is not None:
        mcs = min_cluster_size
    elif n < 60:
        # The big-data floor (15) would mark a fresh interview's whole profile as
        # noise, leaving the seed sampler nothing to wander on.
        mcs = max(2, n // 4)
    else:
        mcs = max(15, n // 80)
    normed = _normalize_rows(embeddings.astype(np.float64))
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw.shape[0] == n:
            # sqrt(weight) row scaling approximates sample_weight in distance (documented trick).
            scale = np.sqrt(np.maximum(sw, 1e-8))[:, None]
            normed = normed * scale

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min(5, mcs),
        metric="euclidean",
    )
    raw_labels = clusterer.fit_predict(normed)
    result = _build_from_labels(
        embeddings, raw_labels.astype(int), sample_weight, "hdbscan"
    )
    if result.n_noise == len(embeddings):
        # Density found no structure at all; an all-noise profile is useless.
        return _cluster_kmeans(embeddings, k=None, sample_weight=sample_weight)
    return result


def _cluster_kmeans(
    embeddings: np.ndarray,
    k: Optional[int],
    sample_weight: Optional[np.ndarray],
) -> ClusterResult:
    """K-means fallback; centroids are cluster centers (nearest-point medoid used)."""
    n = len(embeddings)
    if n < MIN_CLUSTERS:
        labels = np.arange(n)
        return _build_from_labels(embeddings, labels, sample_weight, "kmeans")
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        labels = np.arange(n)
        return _build_from_labels(embeddings, labels, sample_weight, "kmeans")

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

    # Replace K-means centroids with medoids for consistency.
    return _build_from_labels(embeddings, labels.astype(int), sample_weight, "kmeans")


def _cluster_gmm(
    embeddings: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> ClusterResult:
    """Gaussian mixture with BIC k-selection (k = 2 .. min(20, n//10))."""
    n = len(embeddings)
    if n < MIN_CLUSTERS:
        labels = np.arange(n)
        return _build_from_labels(embeddings, labels, sample_weight, "gmm")
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore
    except Exception:
        return _cluster_kmeans(embeddings, k=None, sample_weight=sample_weight)

    max_k = min(20, max(2, n // 10))
    best_k, best_bic, best_labels = 2, float("inf"), None
    sw = None
    if sample_weight is not None:
        sw_arr = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw_arr.shape[0] == n and np.any(sw_arr > 0):
            sw = sw_arr

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, n_init=2, random_state=42)
        if sw is not None:
            gmm.fit(embeddings, sample_weight=sw)
        else:
            gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = gmm.predict(embeddings)

    if best_labels is None:
        return _cluster_kmeans(embeddings, k=None, sample_weight=sample_weight)

    return _build_from_labels(
        embeddings, best_labels.astype(int), sample_weight, "gmm"
    )


def cluster(
    embeddings: np.ndarray,
    k: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
    method: str = "hdbscan",
    recency_weights: Optional[np.ndarray] = None,
    min_cluster_size: Optional[int] = None,
) -> ClusterResult:
    """Cluster embeddings; default method is HDBSCAN on L2-normalized vectors.

    `recency_weights`, when provided, override `sample_weight` (composite recency × visit).
    Falls back to K-means when HDBSCAN is unavailable or method='kmeans'.
    """
    if embeddings is None or len(embeddings) == 0:
        return ClusterResult(
            centroids=np.zeros((0, 0), dtype=np.float32),
            labels=np.array([], dtype=int),
            medoid_indices=[],
            method=method,
            n_noise=0,
            silhouette=None,
        )

    sw = recency_weights if recency_weights is not None else sample_weight
    m = (method or "hdbscan").lower()
    if m == "kmeans":
        return _cluster_kmeans(embeddings, k=k, sample_weight=sw)
    if m == "gmm":
        return _cluster_gmm(embeddings, sample_weight=sw)
    return _cluster_hdbscan(embeddings, sample_weight=sw, min_cluster_size=min_cluster_size)


def pin_manual_interests(
    result: ClusterResult,
    embeddings: np.ndarray,
    sources: Sequence[str],
    sample_weight: Optional[np.ndarray],
    *,
    mega_fraction: float = 0.40,
    group_cosine: float = 0.92,
    weight_boost: float = 3.0,
) -> tuple[ClusterResult, np.ndarray]:
    """Promote manual/interview interests from noise or mega-clusters to pinned mini-clusters.

    Returns updated ClusterResult and boosted sample_weight array.
    """
    labels = result.labels.copy()
    n = len(labels)
    sw = (
        np.asarray(sample_weight, dtype=np.float64).copy()
        if sample_weight is not None
        else np.ones(n, dtype=np.float64)
    )

    mega_label: int | None = None
    for lab in set(int(l) for l in labels if int(l) >= 0):
        count = int(np.sum(labels == lab))
        if count > mega_fraction * n:
            mega_label = lab
            break

    manual_idxs = [
        i
        for i, src in enumerate(sources)
        if src.startswith("manual") or src == "interview"
    ]
    pin_candidates = [
        i
        for i in manual_idxs
        if int(labels[i]) == -1
        or (mega_label is not None and int(labels[i]) == mega_label)
    ]
    if not pin_candidates:
        return result, sw

    # Group similar manual interests (cosine > group_cosine).
    groups: list[list[int]] = []
    used: set[int] = set()
    emb_n = _normalize_rows(embeddings.astype(np.float64))
    for i in pin_candidates:
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in pin_candidates:
            if j in used:
                continue
            if float(np.dot(emb_n[i], emb_n[j])) >= group_cosine:
                group.append(j)
                used.add(j)
        groups.append(group)

    next_label = int(labels.max()) + 1 if int(labels.max()) >= 0 else 0
    for group in groups:
        for idx in group:
            labels[idx] = next_label
            sw[idx] *= weight_boost
        next_label += 1

    rebuilt = _build_from_labels(embeddings, labels, sw, result.method)
    rebuilt.silhouette = result.silhouette
    return rebuilt, sw


def add_noise_cluster(
    result: ClusterResult,
    embeddings: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> ClusterResult:
    """Append an explicit noise-bucket cluster (is_noise) for unclustered points."""
    if result.n_noise <= 0:
        return result
    return _build_from_labels(
        embeddings,
        result.labels,
        sample_weight,
        result.method,
        include_noise_cluster=True,
    )


def dopamine(
    item_emb: np.ndarray,
    centroids: Sequence[np.ndarray],
    recent: Sequence[np.ndarray],
    rng: Optional[random.Random] = None,
    *,
    fulfillment: float = 1.0,
) -> dict:
    """Compute the dopamine reward as a dict with breakdown + total. Logged with every brief."""
    rng = rng or random.Random()
    f = max(0.0, min(1.0, float(fulfillment)))
    a = alignment(item_emb, centroids)
    n = novelty(item_emb, recent)
    s = surprise(item_emb, recent, centroids)
    serendipity = 0.0
    # Epsilon-greedy: small chance to amplify a low-alignment, high-novelty item.
    # Gated on fulfillment so failed searches can't jackpot via serendipity alone.
    if (
        f >= SERENDIPITY_MIN_FULFILLMENT
        and rng.random() < EPS
        and a < 0.4
        and n > 0.6
    ):
        serendipity = 1.0
    total = ALPHA * a + BETA * n + GAMMA * s + DELTA * f + SERENDIPITY_BONUS * serendipity
    return {
        "alignment": round(float(a), 4),
        "novelty": round(float(n), 4),
        "surprise": round(float(s), 4),
        "fulfillment": round(float(f), 4),
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
