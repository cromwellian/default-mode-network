"""Reddit search via the public .json endpoints (no auth)."""
from __future__ import annotations

from . import ResearchItem

USER_AGENT = "default-mode-network/0.1"


def search(
    query: str, max_results: int = 5, subreddit: str | None = None
) -> list[ResearchItem]:
    """Search Reddit (optionally restricted to a subreddit) via public .json endpoints."""
    if not query:
        return []
    try:
        import requests
    except Exception:
        return []
    base = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        if subreddit
        else "https://www.reddit.com/search.json"
    )
    params = {"q": query, "limit": max_results, "sort": "relevance"}
    if subreddit:
        params["restrict_sr"] = "on"
    try:
        r = requests.get(
            base, params=params, headers={"User-Agent": USER_AGENT}, timeout=10
        )
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
    except Exception:
        return []
    out: list[ResearchItem] = []
    for c in children:
        d = c.get("data", {})
        title = d.get("title", "")
        body = (d.get("selftext") or "")[:400] or title
        url = "https://www.reddit.com" + d.get("permalink", "")
        out.append(
            ResearchItem(
                title=title,
                summary=body,
                url=url,
                source="reddit",
                raw_text=d.get("selftext") or "",
            )
        )
    return out
