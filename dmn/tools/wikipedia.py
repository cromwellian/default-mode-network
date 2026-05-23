"""Wikipedia search backend (uses the public Action API with a polite User-Agent)."""
from __future__ import annotations

import re

from . import ResearchItem

USER_AGENT = (
    "default-mode-network/0.1 "
    "(+https://github.com/cromwellian/default-mode-network)"
)


def search(query: str, max_results: int = 3) -> list[ResearchItem]:
    """Look up Wikipedia pages via the public API; returns the search snippet."""
    if not query:
        return []
    try:
        import requests
    except Exception:
        return []
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except Exception:
        return []
    out: list[ResearchItem] = []
    for h in hits:
        title = h.get("title") or ""
        snippet = _strip_html(h.get("snippet") or "")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        out.append(
            ResearchItem(
                title=title,
                summary=snippet,
                url=url,
                source="wikipedia",
                raw_text=snippet,
            )
        )
    return out


def _strip_html(s: str) -> str:
    """Lightweight HTML-tag stripper for Wikipedia snippets."""
    return re.sub(r"<[^>]+>", "", s).replace("&quot;", '"')
