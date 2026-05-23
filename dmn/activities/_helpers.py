"""Small shared helpers for the v0.3 code-generating activities."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Match a fenced code block of a given language at any indent (greedy across newlines).
_CODE_FENCE_TEMPLATE = r"```\s*{lang}\s*\n(.*?)\n\s*```"


def extract_fenced(text: str, lang: str) -> Optional[str]:
    """Return the body of the LAST fenced ```<lang>``` block in text, or None.

    We take the LAST match because the LLM occasionally hallucinates an example
    earlier in the response; the trailing block is usually the answer.
    """
    if not text:
        return None
    pattern = re.compile(_CODE_FENCE_TEMPLATE.format(lang=re.escape(lang)),
                         re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).rstrip()


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
