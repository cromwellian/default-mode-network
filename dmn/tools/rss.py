"""RSS/Atom news feeds (Techmeme + tech/science/culture). Query filters feed items by
keyword overlap; empty query returns the freshest items. Feeds are fetched once per
process (they don't change per query), so grounding many seeds stays cheap."""
from __future__ import annotations

import re
import time
from xml.etree import ElementTree as ET

from . import ResearchItem

# A small curated set spanning tech news, science, and culture. Add more freely.
_FEEDS = {
    "techmeme": "https://www.techmeme.com/feed.xml",
    "theverge": "https://www.theverge.com/rss/index.xml",
    "arstechnica": "https://feeds.arstechnica.com/arstechnica/index",
    "quanta": "https://www.quantamagazine.org/feed/",
}

_CACHE: dict = {"items": None, "ts": 0.0}
_TTL_S = 900  # 15 min — a single wander session reuses one fetch


def _strip(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").replace("&nbsp;", " ").strip()


def _parse(xml_text: str, source: str) -> list[ResearchItem]:
    out: list[ResearchItem] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title = desc = link = ""
        for ch in node:
            t = ch.tag.split("}")[-1]
            if t == "title":
                title = _strip(ch.text)
            elif t in ("description", "summary", "content") and not desc:
                desc = _strip(ch.text)
            elif t == "link" and not link:
                link = ch.get("href") or (ch.text or "")
        if title:
            out.append(
                ResearchItem(
                    title=title,
                    summary=(desc or title)[:300],
                    url=link,
                    source=f"rss:{source}",
                    raw_text=desc,
                )
            )
    return out


def _all_items() -> list[ResearchItem]:
    now = time.time()
    if _CACHE["items"] is not None and (now - _CACHE["ts"]) < _TTL_S:
        return _CACHE["items"]
    try:
        import requests
    except Exception:
        return []
    items: list[ResearchItem] = []
    for name, url in _FEEDS.items():
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent": "default-mode-network"})
            r.raise_for_status()
            items.extend(_parse(r.text, name))
        except Exception:
            continue
    _CACHE["items"] = items
    _CACHE["ts"] = now
    return items


def search(query: str = "", max_results: int = 8) -> list[ResearchItem]:
    items = _all_items()
    if not items:
        return []
    if not query:
        return items[:max_results]
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not terms:
        return items[:max_results]

    def score(it: ResearchItem) -> int:
        text = (it.title + " " + it.summary).lower()
        return sum(text.count(t) for t in terms)

    ranked = sorted(items, key=score, reverse=True)
    hits = [it for it in ranked if score(it) > 0]
    return (hits or items)[:max_results]
