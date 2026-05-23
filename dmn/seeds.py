"""Seed-question generators: cold-start, cluster sampling, cross-pollination, drift, trending.

v0.1.2: every generator that consumes cluster info now reads `cluster.meta.theme` /
`cluster.meta.subtopics` (synthesized by `dmn.labeling`) instead of the raw `cluster.label`,
which used to leak browser-history URLs and YouTube titles into seed prompts.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Seed:
    """A research seed: text, the strategy that produced it, and the subtopic (if any) used."""

    text: str
    source: str  # 'cold' | 'cluster' | 'cross' | 'drift' | 'trending' | 'manual'
    subtopic: Optional[str] = None


def _theme(cluster: dict) -> str:
    """Pull the synthesized theme for a cluster, falling back to label, then a stub string."""
    meta = cluster.get("meta") or {}
    return meta.get("theme") or cluster.get("label") or "this taste"


def _subtopics(cluster: dict) -> list[str]:
    """Return the cluster's subtopics list, or an empty list."""
    meta = cluster.get("meta") or {}
    subs = meta.get("subtopics") or []
    return [s for s in subs if isinstance(s, str) and s.strip()]


def cold_start(clusters: list[dict], rng: random.Random) -> Seed:
    """Pick a random cluster, then a random subtopic if available, else theme. Never raw text."""
    if not clusters:
        return Seed(
            text="What's something the world is currently underestimating?",
            source="cold",
        )
    c = rng.choice(clusters)
    subs = _subtopics(c)
    if subs:
        st = rng.choice(subs)
        return Seed(text=f"What's a fresh angle on {st}?", source="cold", subtopic=st)
    return Seed(text=f"What's a fresh angle on {_theme(c)}?", source="cold")


def cluster_sample(clusters: list[dict], llm, rng: random.Random) -> Seed:
    """Pick a taste cluster, ask the LLM for 5 candidate questions grounded in a subtopic."""
    if not clusters:
        return cold_start([], rng)
    c = rng.choice(clusters)
    theme = _theme(c)
    subs = _subtopics(c)
    subtopic = rng.choice(subs) if subs else ""
    qualifier = f" Specifically around: {subtopic}." if subtopic else ""
    prompt = (
        f"My friend is curious about themes around '{theme}'.{qualifier} "
        "Brainstorm 5 novel research questions in a numbered list. Be concrete, not abstract."
    )
    try:
        resp = llm.complete(
            system="You are a curious librarian.", user=prompt, max_tokens=400
        )
        qs = _extract_numbered(resp.text)
    except Exception:
        qs = []
    text = (
        rng.choice(qs)
        if qs
        else f"What's an unanswered question in '{theme}'?"
    )
    return Seed(text=text, source="cluster", subtopic=subtopic or None)


def cross_pollinate(clusters: list[dict], llm, rng: random.Random) -> Seed:
    """Pick the two MOST DISTANT clusters by min cosine sim; brainstorm at the intersection."""
    if len(clusters) < 2:
        return cluster_sample(clusters, llm, rng)
    a, b = _most_distant_pair(clusters, rng)
    la, lb = _theme(a), _theme(b)
    sa = rng.choice(_subtopics(a)) if _subtopics(a) else ""
    sb = rng.choice(_subtopics(b)) if _subtopics(b) else ""
    sub_clauses = [s for s in (sa, sb) if s]
    sep = " \u2194 "
    qualifier = f" Specifically: {sep.join(sub_clauses)}." if sub_clauses else ""
    prompt = (
        f"Brainstorm 3 surprising research questions that sit at the intersection of "
        f"'{la}' and '{lb}'.{qualifier} Numbered list. Concrete, not abstract."
    )
    try:
        resp = llm.complete(
            system="You connect distant ideas.", user=prompt, max_tokens=400
        )
        qs = _extract_numbered(resp.text)
    except Exception:
        qs = []
    text = (
        rng.choice(qs)
        if qs
        else f"How do '{la}' and '{lb}' speak to each other?"
    )
    return Seed(text=text, source="cross", subtopic=(sa or sb) or None)


def _most_distant_pair(
    clusters: list[dict], rng: random.Random
) -> tuple[dict, dict]:
    """Return the two clusters with minimum cosine similarity between centroids."""
    import numpy as np

    valid = [c for c in clusters if c.get("centroid") is not None]
    if len(valid) < 2:
        return tuple(rng.sample(clusters, 2))  # type: ignore[return-value]
    cents = [np.asarray(c["centroid"], dtype=np.float32) for c in valid]
    norms = [c / (np.linalg.norm(c) + 1e-8) for c in cents]
    min_sim = float("inf")
    best: tuple[int, int] | None = None
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            sim = float(np.dot(norms[i], norms[j]))
            if sim < min_sim:
                min_sim = sim
                best = (i, j)
    if best is None:
        return tuple(rng.sample(clusters, 2))  # type: ignore[return-value]
    return valid[best[0]], valid[best[1]]


def drift(recent_briefs: list[dict], llm, rng: random.Random) -> Seed:
    """Take a recent journal entry and drift one step sideways."""
    if not recent_briefs:
        return Seed(text="What's a thread I haven't pulled yet today?", source="drift")
    brief = rng.choice(recent_briefs[: min(5, len(recent_briefs))])
    seed_text = brief.get("seed") or ""
    prompt = (
        f"Given the recent question '{seed_text}', what's ONE adjacent question, "
        "exactly one step sideways? Reply with just the question."
    )
    try:
        resp = llm.complete(
            system="You drift sideways from ideas.", user=prompt, max_tokens=200
        )
        text = resp.text.strip().split("\n")[0]
    except Exception:
        text = ""
    return Seed(text=text or f"Adjacent to: {seed_text}", source="drift")


def trending_meets_taste(centroids, embed_fn, rng: random.Random) -> Optional[Seed]:
    """Pull a list of trending HN items, score against the taste profile, return the best."""
    try:
        from dmn.tools.hackernews import search as hn_search
    except Exception:
        return None
    items = []
    try:
        items = hn_search("") or []
    except Exception:
        items = []
    if not items or centroids is None or len(centroids) == 0:
        return None
    import numpy as np

    titles = [it.title for it in items[:50] if it.title]
    if not titles:
        return None
    embs = embed_fn(titles)
    cents = np.stack([np.asarray(c, dtype=np.float32) for c in centroids])
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    cents_n = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    sims = (embs_n @ cents_n.T).max(axis=1)
    best_idx = int(sims.argmax())
    chosen = items[best_idx]
    return Seed(
        text=f"What is the deeper story behind: {chosen.title}", source="trending"
    )


def _extract_numbered(text: str) -> list[str]:
    """Pull numbered-list items out of an LLM response."""
    out: list[str] = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            out.append(m.group(1).strip())
    return out
