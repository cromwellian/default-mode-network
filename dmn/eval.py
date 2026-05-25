"""User-rating evaluation for the dopamine reward.

The reward function is intentionally soft, so it needs a feedback loop. This module keeps
that loop small: users rate briefs 1..5, then we compare those ratings to the logged
dopamine components.
"""
from __future__ import annotations

import math
from typing import Iterable


DOPAMINE_KEYS = [
    "alignment",
    "novelty",
    "surprise",
    "fulfillment",
    "serendipity",
    "total",
]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson correlation, or None when there is not enough variance."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def summarize(entries: Iterable[dict]) -> dict:
    """Summarize rated journal entries and component correlations."""
    rows = [e for e in entries if e.get("user_rating") is not None]
    ratings = [float(e["user_rating"]) for e in rows]
    correlations: dict[str, float | None] = {}
    for key in DOPAMINE_KEYS:
        xs: list[float] = []
        ys: list[float] = []
        for e in rows:
            breakdown = e.get("dopamine_breakdown") or {}
            value = breakdown.get(key)
            if value is None and key == "total":
                value = e.get("dopamine_total")
            if value is None:
                continue
            xs.append(float(value))
            ys.append(float(e["user_rating"]))
        correlations[key] = pearson(xs, ys)
    return {
        "count": len(rows),
        "mean_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "correlations": correlations,
        "rated": rows,
    }


def suggestions(summary: dict) -> list[str]:
    """Small deterministic tuning hints from a rating summary."""
    if summary.get("count", 0) < 5:
        return ["Collect at least 5 ratings before tuning dopamine weights."]
    cors = summary.get("correlations") or {}
    out: list[str] = []
    total = cors.get("total")
    if total is not None and total < 0.2:
        out.append("Total dopamine is weakly aligned with ratings; inspect component weights.")
    for key in ("alignment", "novelty", "surprise", "fulfillment"):
        c = cors.get(key)
        if c is not None and c < -0.1:
            out.append(f"`{key}` is negatively correlated with ratings; consider lowering its weight.")
        elif c is not None and c > 0.35:
            out.append(f"`{key}` tracks ratings well; it may deserve more weight.")
    return out or ["No obvious tuning change yet; keep collecting ratings."]
