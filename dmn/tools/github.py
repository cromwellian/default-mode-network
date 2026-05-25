"""GitHub repository search (no key needed). Query → relevant repos by stars; empty query
→ repos created in the last 30 days, most-starred (a 'what's rising' proxy)."""
from __future__ import annotations

from . import ResearchItem

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "default-mode-network",
}


def search(query: str = "", max_results: int = 8) -> list[ResearchItem]:
    try:
        import requests
    except Exception:
        return []
    try:
        if query:
            params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}
        else:
            from datetime import date, timedelta

            since = (date.today() - timedelta(days=30)).isoformat()
            params = {
                "q": f"created:>{since}",
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }
        r = requests.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception:
        return []
    out: list[ResearchItem] = []
    for it in items[:max_results]:
        desc = (it.get("description") or "").strip()
        stars = it.get("stargazers_count", 0)
        lang = it.get("language") or ""
        tail = f"★{stars}" + (f", {lang}" if lang else "")
        out.append(
            ResearchItem(
                title=it.get("full_name") or it.get("name") or "",
                summary=(f"{desc} ({tail})" if desc else tail) or "github repo",
                url=it.get("html_url") or "",
                source="github",
                raw_text=desc,
            )
        )
    return out
