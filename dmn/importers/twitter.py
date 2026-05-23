"""Twitter/X export likes.js importer."""
from __future__ import annotations

import json
import re
from pathlib import Path


def import_likes(export_dir: str | Path, limit: int = 500) -> list[dict]:
    """Parse like.js / likes.js out of a Twitter/X data export."""
    base = Path(export_dir).expanduser()
    if not base.exists():
        return []
    candidates = list(base.rglob("like.js")) + list(base.rglob("likes.js"))
    if not candidates:
        return []
    text = candidates[0].read_text(errors="ignore")
    text = re.sub(r"^\s*window\.YTD\.[a-zA-Z\.]+\s*=\s*", "", text)
    try:
        data = json.loads(text)
    except Exception:
        return []
    out: list[dict] = []
    for entry in data[:limit]:
        like = entry.get("like", entry) if isinstance(entry, dict) else {}
        full = like.get("fullText") or like.get("expandedUrl") or ""
        if full:
            out.append({"text": full[:500], "source": "twitter"})
    return out
