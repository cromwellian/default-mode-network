"""Google Drive Takeout importer: walk the Drive/ tree and pull doc titles (filenames)."""
from __future__ import annotations

from pathlib import Path

DOC_SUFFIXES = {
    ".gdoc",
    ".gsheet",
    ".gslides",
    ".pdf",
    ".md",
    ".txt",
    ".docx",
    ".rtf",
}


def import_drive_titles(takeout_dir: str | Path, limit: int = 500) -> list[dict]:
    """Walk Takeout/Drive and collect doc filenames (no body text — titles only)."""
    base = Path(takeout_dir).expanduser()
    drive_root = base / "Drive"
    root = drive_root if drive_root.exists() else base
    if not root.exists():
        return []
    out: list[dict] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in DOC_SUFFIXES:
            continue
        out.append({"text": p.stem, "source": "drive"})
        if len(out) >= limit:
            break
    return out
