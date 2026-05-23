"""YouTube Data API search; gracefully no-ops without YOUTUBE_API_KEY."""
from __future__ import annotations

import os

from . import ResearchItem


def search(query: str, max_results: int = 5) -> list[ResearchItem]:
    """Search YouTube. Returns [] if YOUTUBE_API_KEY is not set or any error occurs."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key or not query:
        return []
    try:
        import requests
    except Exception:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": query,
                "part": "snippet",
                "maxResults": max_results,
                "type": "video",
                "key": key,
            },
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items", []) or []
    except Exception:
        return []
    out: list[ResearchItem] = []
    for it in items:
        sn = it.get("snippet", {}) or {}
        vid = (it.get("id") or {}).get("videoId") or ""
        out.append(
            ResearchItem(
                title=sn.get("title") or "",
                summary=sn.get("description") or "",
                url=f"https://www.youtube.com/watch?v={vid}" if vid else "",
                source="youtube",
                raw_text=sn.get("description") or "",
            )
        )
    return out
