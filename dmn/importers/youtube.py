"""Google Takeout YouTube watch-history importer (.json or .html)."""
from __future__ import annotations

import json
import re
from pathlib import Path


def import_watch_history(takeout_dir: str | Path, limit: int = 500) -> list[dict]:
    """Locate watch-history.{json,html} under a Takeout dir and parse titles."""
    base = Path(takeout_dir).expanduser()
    if not base.exists():
        return []
    candidates = list(base.rglob("watch-history.json")) + list(
        base.rglob("watch-history.html")
    )
    if not candidates:
        return []
    p = candidates[0]
    if p.suffix == ".json":
        return _parse_json(p, limit)
    return _parse_html(p, limit)


def _parse_json(p: Path, limit: int) -> list[dict]:
    """Parse a Takeout watch-history.json file."""
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    out: list[dict] = []
    for entry in data[:limit]:
        title = entry.get("title") or entry.get("header") or ""
        if title.startswith("Watched "):
            title = title[len("Watched ") :]
        url = entry.get("titleUrl") or ""
        if title:
            out.append(
                {"text": f"{title} ({url})".strip(" ()"), "source": "youtube"}
            )
    return out


def _parse_html(p: Path, limit: int) -> list[dict]:
    """Parse a Takeout watch-history.html file using a simple regex over the title cells."""
    text = p.read_text(errors="ignore")
    titles = re.findall(r">Watched ([^<]+)<", text)
    return [{"text": t, "source": "youtube"} for t in titles[:limit]]
