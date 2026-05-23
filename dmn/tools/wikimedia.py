"""Wikimedia Commons image search (no key needed)."""
from __future__ import annotations

from . import ResearchItem


def search(query: str, max_results: int = 5) -> list[ResearchItem]:
    """Search Wikimedia Commons for image files matching a query."""
    if not query:
        return []
    try:
        import requests
    except Exception:
        return []
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 6,
                "format": "json",
                "srlimit": max_results,
            },
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except Exception:
        return []
    out: list[ResearchItem] = []
    for h in hits:
        title = h.get("title") or ""
        out.append(
            ResearchItem(
                title=title,
                summary=h.get("snippet") or "",
                url=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                source="wikimedia",
                raw_text=h.get("snippet") or "",
            )
        )
    return out
