"""Fulfillment scoring: did research/tools actually return useful material?"""
from __future__ import annotations

import re
from typing import Sequence

from dmn import taste

# LLM self-report phrases indicating the search whiffed.
_META_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"search whiffed",
        r"don'?t actually contain",
        r"search failure",
        r"didn'?t return relevant",
        r"struck out",
        r"instead of market analysis",
        r"no relevant results",
        r"nothing useful",
        r"failed to find",
        r"couldn'?t find (?:any )?relevant",
    )
]

META_FAILURE_MULTIPLIER = 0.4
RELEVANCE_ALIGNMENT_THRESHOLD = 0.15
RELEVANCE_CAP = 0.2


def _results_count_score(n: int) -> float:
    if n <= 0:
        return 0.0
    if n <= 2:
        return 0.3
    if n <= 4:
        return 0.7
    return 1.0


def _detect_meta_failure(body_md: str) -> bool:
    text = body_md or ""
    return any(p.search(text) for p in _META_FAILURE_PATTERNS)


def compute_fulfillment(
    *,
    tool_items: list,
    item_scores: Sequence[float],
    item_alignments: Sequence[float] | None = None,
    body_md: str = "",
    activity: str = "research",
    has_artifact: bool = False,
) -> tuple[float, dict]:
    """Return (fulfillment 0..1, breakdown dict).

    For research: count + quality of tool results, relevance cap, meta-failure penalty.
    For other activities: 1.0 if artifact produced, 0.5 if body only, 0.0 if empty.
    """
    if activity != "research":
        has_body = bool((body_md or "").strip())
        if has_artifact:
            score = 1.0
        elif has_body:
            score = 0.5
        else:
            score = 0.0
        return score, {
            "results_count": 1.0 if has_artifact else 0.0,
            "results_quality": 1.0 if has_body or has_artifact else 0.0,
            "relevance": 1.0,
            "meta_failure": 0.0,
            "fulfillment": score,
        }

    n = len(tool_items)
    count_score = _results_count_score(n)

    alignments = list(item_alignments or [])
    if not alignments and item_scores:
        # Fall back to item dopamine totals, normalized to ~0..1 range.
        top = sorted(float(s) for s in item_scores)[:3]
        alignments = [min(1.0, max(0.0, s)) for s in top]

    quality_score = float(sum(alignments[:3]) / len(alignments[:3])) if alignments else 0.0

    raw = 0.4 * count_score + 0.6 * quality_score
    mean_alignment = float(sum(alignments) / len(alignments)) if alignments else 0.0
    relevance = 1.0
    if alignments and mean_alignment < RELEVANCE_ALIGNMENT_THRESHOLD:
        raw = min(raw, RELEVANCE_CAP)
        relevance = 0.0

    meta_failure = 1.0 if _detect_meta_failure(body_md) else 0.0
    fulfillment = raw * (META_FAILURE_MULTIPLIER if meta_failure else 1.0)

    breakdown = {
        "results_count": round(count_score, 4),
        "results_quality": round(quality_score, 4),
        "relevance": round(relevance, 4),
        "meta_failure": round(meta_failure, 4),
        "fulfillment": round(float(fulfillment), 4),
    }
    return float(fulfillment), breakdown


def item_alignments(
    items: list,
    centroids: Sequence,
) -> list[float]:
    """Per-item alignment scores for fulfillment quality/relevance."""
    out: list[float] = []
    for it in items:
        emb = getattr(it, "embedding", None)
        if emb is None:
            out.append(0.0)
        else:
            out.append(taste.alignment(emb, centroids))
    return out
