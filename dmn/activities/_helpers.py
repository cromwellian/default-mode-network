"""Small shared helpers for the v0.3 code-generating activities."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Match a fenced code block of a given language at any indent (greedy across newlines).
_CODE_FENCE_TEMPLATE = r"```\s*{lang}\s*\n(.*?)\n\s*```"


def extract_fenced(text: str, lang: str, *, salvage: bool = False) -> Optional[str]:
    """Return the body of the LAST fenced ```<lang>``` block in text, or None.

    We take the LAST match because the LLM occasionally hallucinates an example
    earlier in the response; the trailing block is usually the answer.

    With `salvage=True`, if no *closed* fence is found we recover an unclosed one — the
    common case when the response was truncated at max_tokens before the closing ```.
    Without this, a truncated code/HTML block silently yields None and callers fall back
    to a canned stub (the "always the same dry-run app" bug).
    """
    if not text:
        return None
    pattern = re.compile(_CODE_FENCE_TEMPLATE.format(lang=re.escape(lang)),
                         re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1).rstrip()
    if not salvage:
        return None
    open_re = re.compile(r"```\s*" + re.escape(lang) + r"\s*\n", re.IGNORECASE)
    opens = list(open_re.finditer(text))
    if not opens:
        return None
    body = text[opens[-1].end():]
    # Strip a dangling/partial closing fence if the truncation left one.
    body = re.sub(r"\n\s*```.*$", "", body, flags=re.DOTALL).rstrip()
    return body or None


def extract_html(text: str) -> Optional[str]:
    """Extract an HTML document from an LLM response, tolerating truncation.

    Tries a ```html fence (closed or unclosed), then a bare <!doctype>/<html> document.
    Returns None when there's nothing HTML-ish to salvage.
    """
    body = extract_fenced(text, "html", salvage=True)
    if body and "<" in body:
        return body
    if text:
        m = re.search(r"(?is)(<!doctype html\b.*|<html[\s>].*)", text)
        if m:
            return m.group(1).rstrip()
    return None


def extract_json_meta(text: str) -> dict:
    """Extract the trailing ```json metadata block, returning {} on any parse failure."""
    body = extract_fenced(text, "json")
    if body is None:
        return {}
    try:
        data = json.loads(body)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def slugify(text: str, n: int = 60) -> str:
    """Filesystem-safe slug from arbitrary text."""
    s = re.sub(r"[^a-zA-Z0-9\-_ ]", "", text or "").strip().lower().replace(" ", "-")
    return (s[:n] or "brief").strip("-") or "brief"


def write_artifact_dir(root: Path, activity_name: str, slug: str) -> Path:
    """Create + return data/artifacts/<activity>/<slug>/ for an activity's outputs."""
    out = Path(root) / activity_name / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def llm_first_line(llm, system: str, user: str, max_tokens: int = 120) -> str:
    """Prompt the LLM and return its first non-empty stripped line."""
    try:
        resp = llm.complete(system=system, user=user, max_tokens=max_tokens)
        text = (resp.text or "").strip()
    except Exception:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text
