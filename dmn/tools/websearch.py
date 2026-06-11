"""DuckDuckGo web search (the `duckduckgo-search` package has had API churn; we degrade defensively)."""
from __future__ import annotations

from . import ResearchItem


def search(query: str, max_results: int = 5) -> list[ResearchItem]:
    """DuckDuckGo text search; returns [] on any failure or missing dependency."""
    if not query:
        return []
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return []
    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            try:
                results = list(ddgs.text(query, max_results=max_results))
            except TypeError:
                results = list(ddgs.text(query))[:max_results]
    except Exception:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query[:80]))[:max_results]
        except Exception:
            return []
    out: list[ResearchItem] = []
    for r in results:
        title = r.get("title") or r.get("heading") or ""
        body = r.get("body") or r.get("snippet") or ""
        url = r.get("href") or r.get("url") or ""
        out.append(
            ResearchItem(
                title=title,
                summary=body,
                url=url,
                source="duckduckgo",
                raw_text=body,
            )
        )
    return out
