"""Serialize wander results for REST API responses."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from dmn.api_paths import artifact_url
from dmn.wander_config import WanderResult

_TEXT_SUFFIXES = {".md", ".html", ".json", ".txt", ".py", ".csv", ".xml", ".yaml", ".yml"}
_INLINE_LIMIT = 256_000  # bytes; larger binaries are URL-only


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "image"
    if suffix in {".mp3", ".wav", ".ogg", ".m4a"}:
        return "audio"
    if suffix in {".mp4", ".webm", ".mov"}:
        return "video"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".md"}:
        return "markdown"
    if suffix in {".json"}:
        return "json"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    return "binary"


def _maybe_inline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    size = path.stat().st_size
    kind = _artifact_kind(path)
    payload: dict[str, Any] = {
        "size_bytes": size,
        "mime_type": mimetypes.guess_type(path.name)[0],
    }
    if kind in {"text", "markdown", "html", "json"} and size <= _INLINE_LIMIT:
        payload["content"] = path.read_text(encoding="utf-8", errors="replace")
        return payload
    if kind == "image" and size <= _INLINE_LIMIT:
        raw = path.read_bytes()
        mime = payload["mime_type"] or "application/octet-stream"
        payload["content_base64"] = base64.b64encode(raw).decode("ascii")
        payload["encoding"] = "base64"
        return payload
    return {"size_bytes": size, "mime_type": payload["mime_type"]}


def build_wander_response(
    result: WanderResult,
    *,
    user_id: str,
    profile_id: str,
    base_url: str,
) -> dict[str, Any]:
    """Build JSON payload with inline content (when small) and fetch URLs for all files."""
    work_dir = Path(result.work_dir)
    artifacts: list[dict[str, Any]] = []
    urls: dict[str, str] = {}

    for rel_path, abs_path in sorted(result.files.items()):
        p = Path(abs_path)
        url = artifact_url(base_url, user_id, profile_id, result.run_id, rel_path)
        urls[rel_path] = url
        entry: dict[str, Any] = {
            "path": rel_path,
            "url": url,
            "kind": _artifact_kind(p),
        }
        inline = _maybe_inline(p)
        if inline:
            entry.update(inline)
        artifacts.append(entry)

    return {
        "user_id": user_id,
        "profile_id": profile_id,
        "run_id": result.run_id,
        "status": "completed",
        "stats": {
            "brief_count": result.brief_count,
            "best_score": result.best_score,
            "pruned_count": result.pruned_count,
            "leaf_count": result.leaf_count,
            "open_count": result.open_count,
            "duration_seconds": result.duration_seconds,
            "patience_triggered": result.patience_triggered,
            "beam_pruned": result.beam_pruned,
        },
        "briefs": result.briefs,
        "report_path": result.report_path,
        "report_url": (
            artifact_url(base_url, user_id, profile_id, result.run_id, result.report_path)
            if result.report_path
            else None
        ),
        "tree_path": result.tree_path,
        "tree_url": (
            artifact_url(base_url, user_id, profile_id, result.run_id, result.tree_path)
            if result.tree_path
            else None
        ),
        "artifacts": artifacts,
        "urls": urls,
    }
