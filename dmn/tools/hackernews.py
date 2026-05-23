"""Hacker News search via the Algolia API (no key needed; empty query returns front page)."""
from __future__ import annotations

from . import ResearchItem


def search(query: str = "", max_results: int = 10) -> list[ResearchItem]:
    """Algolia HN search. Empty query returns the current front page."""
    try:
        import requests
    except Exception:
        return []
    try:
        if query:
            params = {"query": query, "tags": "story", "hitsPerPage": max_results}
        else:
            params = {"tags": "front_page", "hitsPerPage": max_results}
        r = requests.get(
            "https://hn.algolia.com/api/v1/search", params=params, timeout=10
        )
        r.raise_for_status()
        hits = r.json().get("hits", [])
    except Exception:
        return []
    out: list[ResearchItem] = []
    for h in hits:
        title = h.get("title") or h.get("story_title") or ""
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        body = (h.get("story_text") or "")[:500]
        out.append(
            ResearchItem(
                title=title,
                summary=body or title,
                url=url,
                source="hackernews",
                raw_text=body,
            )
        )
    return out
