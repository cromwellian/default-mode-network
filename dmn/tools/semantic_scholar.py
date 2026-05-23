"""Semantic Scholar paper search (no key needed for low volume)."""
from __future__ import annotations

from . import ResearchItem


def search(query: str, max_results: int = 5) -> list[ResearchItem]:
    """Search Semantic Scholar's public API; degrades to [] on any error."""
    if not query:
        return []
    try:
        import requests
    except Exception:
        return []
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,url,authors,year",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("data", []) or []
    except Exception:
        return []
    out: list[ResearchItem] = []
    for p in data:
        out.append(
            ResearchItem(
                title=p.get("title") or "",
                summary=(p.get("abstract") or "")[:600],
                url=p.get("url") or "",
                source="semantic_scholar",
                raw_text=p.get("abstract") or "",
            )
        )
    return out
