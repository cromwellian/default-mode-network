"""ArXiv search backend (uses the official `arxiv` python package; no key needed)."""
from __future__ import annotations

from . import ResearchItem

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "stat.ML", "cs.CL", "q-bio", "physics"]


def search(
    query: str,
    max_results: int = 5,
    categories: list[str] | None = None,
) -> list[ResearchItem]:
    """Search ArXiv; returns [] gracefully if the `arxiv` package is missing."""
    try:
        import arxiv  # type: ignore
    except Exception:
        return []
    cats = categories or DEFAULT_CATEGORIES
    cat_filter = " OR ".join(f"cat:{c}" for c in cats)
    search_q = f"({query}) AND ({cat_filter})" if query else cat_filter
    out: list[ResearchItem] = []
    try:
        client = arxiv.Client(page_size=max_results, num_retries=2, delay_seconds=1)
        gen = client.results(arxiv.Search(query=search_q, max_results=max_results))
        for r in gen:
            out.append(
                ResearchItem(
                    title=(r.title or "").strip(),
                    summary=(r.summary or "").strip()[:600],
                    url=r.entry_id,
                    source="arxiv",
                    raw_text=r.summary or "",
                )
            )
    except Exception:
        return out
    return out
