"""Path helpers for markdown artifacts embedded in journal briefs."""
from __future__ import annotations

import os
from pathlib import Path


def artifact_href(brief_path: Path | str, artifact_path: Path | str) -> str:
    """Return a markdown-safe href from a brief file to an on-disk artifact.

    Briefs live under ``journal/``; artifacts under ``data/artifacts/``. Cursor/VS Code
    markdown preview resolves relative paths from the brief file, not the repo root.
    """
    target = str(artifact_path).strip()
    if not target:
        return ""
    if target.startswith(("http://", "https://", "data:")):
        return target

    art = Path(target)
    if art.is_absolute():
        try:
            art = art.relative_to(Path.cwd())
        except ValueError:
            return art.as_posix()

    brief_dir = Path(brief_path).parent.resolve()
    art_resolved = (Path.cwd() / art).resolve()
    return Path(os.path.relpath(art_resolved, brief_dir)).as_posix()


def artifact_href_for_journal(
    journal_dir: Path | str, artifact_path: Path | str
) -> str:
    """Href as if the brief lives directly under ``journal_dir`` (flat brief layout)."""
    return artifact_href(Path(journal_dir) / "_brief.md", artifact_path)
