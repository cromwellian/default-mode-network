"""MusicBrainz recording lookup (no key needed; rate-limited at 1 req/s)."""
from __future__ import annotations

import time

from . import ResearchItem


def search(query: str, max_results: int = 5) -> list[ResearchItem]:
    """Search MusicBrainz recordings; gracefully degrades on missing requests."""
    if not query:
        return []
    try:
        import requests
    except Exception:
        return []
    try:
        time.sleep(0.2)
        r = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params={"query": query, "limit": max_results, "fmt": "json"},
            headers={
                "User-Agent": "default-mode-network/0.1 (mailto:contact@example.com)"
            },
            timeout=10,
        )
        r.raise_for_status()
        recs = r.json().get("recordings", []) or []
    except Exception:
        return []
    out: list[ResearchItem] = []
    for rec in recs:
        artist = ", ".join(
            a.get("name", "")
            for a in rec.get("artist-credit", []) or []
            if isinstance(a, dict)
        )
        title = rec.get("title", "")
        out.append(
            ResearchItem(
                title=f"{artist} — {title}" if artist else title,
                summary=f"Length: {rec.get('length')}; tags: "
                f"{[t.get('name') for t in (rec.get('tags') or [])]}",
                url=f"https://musicbrainz.org/recording/{rec.get('id', '')}",
                source="musicbrainz",
                raw_text="",
            )
        )
    return out
