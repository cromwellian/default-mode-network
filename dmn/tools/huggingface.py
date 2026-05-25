"""Hugging Face model search (public API; HF_TOKEN used if present). Surfaces new/popular
models for a topic — the 'new ML model announcements' source."""
from __future__ import annotations

import os

from . import ResearchItem


def _headers() -> dict:
    h = {"User-Agent": "default-mode-network"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def search(query: str = "", max_results: int = 8) -> list[ResearchItem]:
    try:
        import requests
    except Exception:
        return []
    params: dict = {"sort": "likes", "direction": -1, "limit": max_results}
    if query:
        params["search"] = query
    try:
        r = requests.get(
            "https://huggingface.co/api/models",
            params=params,
            headers=_headers(),
            timeout=10,
        )
        r.raise_for_status()
        models = r.json()
    except Exception:
        return []
    out: list[ResearchItem] = []
    for m in models[:max_results]:
        mid = m.get("id") or m.get("modelId") or ""
        if not mid:
            continue
        likes = m.get("likes", 0)
        downloads = m.get("downloads", 0)
        tags = ", ".join((m.get("tags") or [])[:4])
        out.append(
            ResearchItem(
                title=mid,
                summary=f"HF model · ♥{likes} · ⬇{downloads}" + (f" · {tags}" if tags else ""),
                url=f"https://huggingface.co/{mid}",
                source="huggingface",
                raw_text=tags,
            )
        )
    return out
