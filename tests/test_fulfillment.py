"""Tests for fulfillment scoring and dopamine gating."""
from __future__ import annotations

import numpy as np

from dmn import taste
from dmn.fulfillment import compute_fulfillment


def test_empty_items_zero_fulfillment():
    f, breakdown = compute_fulfillment(
        tool_items=[],
        item_scores=[],
        item_alignments=[],
        activity="research",
    )
    assert f == 0.0
    assert breakdown["results_count"] == 0.0
    assert breakdown["fulfillment"] == 0.0


def test_good_items_high_fulfillment():
    f, breakdown = compute_fulfillment(
        tool_items=[1, 2, 3, 4, 5],
        item_scores=[0.8, 0.7, 0.6, 0.5, 0.4],
        item_alignments=[0.75, 0.70, 0.65, 0.60, 0.55],
        activity="research",
    )
    assert f >= 0.7
    assert breakdown["results_count"] == 1.0
    assert breakdown["results_quality"] >= 0.65


def test_meta_failure_penalty():
    base_body = "Some findings about quantization."
    f_ok, _ = compute_fulfillment(
        tool_items=[1, 2],
        item_scores=[0.2, 0.15],
        item_alignments=[0.2, 0.15],
        body_md=base_body,
        activity="research",
    )
    f_fail, breakdown = compute_fulfillment(
        tool_items=[1, 2],
        item_scores=[0.2, 0.15],
        item_alignments=[0.2, 0.15],
        body_md=base_body + " The search whiffed — results don't actually contain market analysis.",
        activity="research",
    )
    assert breakdown["meta_failure"] == 1.0
    assert f_fail < f_ok
    assert f_fail <= f_ok * 0.41


def test_serendipity_blocked_when_fulfillment_low():
    rng = __import__("random").Random(0)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    centroids = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    recent = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]

    hits = 0
    for seed in range(200):
        r = __import__("random").Random(seed)
        d = taste.dopamine(emb, centroids, recent, r, fulfillment=0.1)
        hits += d["serendipity"]
    assert hits == 0


def test_serendipity_can_fire_when_fulfillment_high():
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    centroids = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    recent = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]

    hits = 0
    for seed in range(500):
        r = __import__("random").Random(seed)
        d = taste.dopamine(emb, centroids, recent, r, fulfillment=1.0)
        hits += d["serendipity"]
    assert hits > 0


def test_failed_search_brief_scores_below_leaderboard():
    """Simulate brief 2026-05-23-230418: high novelty, low alignment, serendipity jackpot."""
    rng = __import__("random").Random(42)
    # Orthogonal to taste centroid and recent → high novelty, near-zero alignment.
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    centroids = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    recent = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]

    a = taste.alignment(emb, centroids)
    n = taste.novelty(emb, recent)
    s = taste.surprise(emb, recent, centroids)
    old_total = 0.55 * a + 0.25 * n + 0.15 * s + 0.05 * 1.0

    f, _ = compute_fulfillment(
        tool_items=[1, 2],
        item_scores=[0.05, 0.04],
        item_alignments=[0.05, 0.04],
        body_md="The search whiffed — these pages don't actually contain Lloyd-Max analysis.",
        activity="research",
    )
    new = taste.dopamine(emb, centroids, recent, rng, fulfillment=f)

    assert a < 0.05
    assert n > 0.9
    assert f < 0.15
    assert new["serendipity"] == 0.0
    assert new["total"] < 0.25
    assert new["total"] < old_total - 0.05
