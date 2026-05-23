"""Readwise / Pocket / GoodReads CSV importer (any well-formed CSV; we look for common columns)."""
from __future__ import annotations

import csv
from pathlib import Path


def import_csv(path: str | Path, limit: int = 500) -> list[dict]:
    """Import a generic CSV; preferred columns are Title / title / Highlight / text."""
    p = Path(path).expanduser()
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            text = (
                row.get("Title")
                or row.get("title")
                or row.get("Highlight")
                or row.get("highlight")
                or row.get("text")
                or row.get("Note")
                or ""
            )
            if not text:
                text = " :: ".join(
                    str(v) for v in row.values() if v is not None and str(v)
                )[:300]
            text = (text or "").strip()
            if text:
                out.append({"text": text, "source": "readwise"})
    return out
