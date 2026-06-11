"""Tests for HDBSCAN clustering on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from dmn import taste


def _make_blobs_with_noise(
    n_per_blob: int = 40,
    noise: int = 30,
    dim: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Three separated blobs plus uniform random noise."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    parts = []
    true_labels = []
    for k, c in enumerate(centers):
        pts = c + 0.05 * rng.standard_normal((n_per_blob, dim))
        parts.append(pts)
        true_labels.extend([k] * n_per_blob)
    noise_pts = rng.standard_normal((noise, dim))
    parts.append(noise_pts)
    true_labels.extend([-1] * noise)
    return np.vstack(parts).astype(np.float32), np.array(true_labels, dtype=int)


hdbscan = pytest.importorskip("hdbscan")


def test_hdbscan_finds_three_blobs_and_noise():
    """HDBSCAN should recover ~3 clusters and separate noise on synthetic blobs."""
    embeddings, _true = _make_blobs_with_noise()
    result = taste.cluster(embeddings, method="hdbscan", min_cluster_size=15)

    non_noise = result.labels[result.labels >= 0]
    n_clusters = len(set(int(l) for l in non_noise))
    assert n_clusters >= 2, f"expected multiple clusters, got {n_clusters}"
    assert result.n_noise >= 5, f"expected noise separation, got n_noise={result.n_noise}"
    assert len(result.medoid_indices) == len(result.centroids)
    assert result.method == "hdbscan"


def test_kmeans_fallback_on_tiny_input():
    """Tiny inputs fall back to one cluster per point."""
    emb = np.random.randn(3, 8).astype(np.float32)
    result = taste.cluster(emb, method="kmeans")
    assert len(result.centroids) == 3
    assert result.method == "kmeans"


def test_composite_weight_manual_beats_old_browser():
    """Manual sources get higher composite weight than stale browser rows."""
    manual_w = taste.composite_weight(5, 365.0, "manual")
    interview_w = taste.composite_weight(5, 365.0, "interview")
    browser_w = taste.composite_weight(5, 365.0, "browser:chrome")
    assert manual_w > browser_w
    assert interview_w > browser_w
    assert taste.composite_weight(10, 0.0, "browser:chrome") > browser_w


def test_recency_decay():
    """Recency halves every half-life period."""
    assert taste.recency_decay(0.0) == pytest.approx(1.0)
    assert taste.recency_decay(90.0) == pytest.approx(0.5)
    assert taste.recency_decay(180.0) == pytest.approx(0.25)


def test_small_profile_never_ends_up_all_noise():
    """A fresh interview (~12 interests) must yield usable clusters, not 100% noise (issue #7)."""
    import numpy as np

    from dmn import taste

    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((12, 64)).astype(np.float32)
    result = taste.cluster(vecs, method="hdbscan")
    assert result.n_noise < len(vecs)
    assert len(result.centroids) >= 2
