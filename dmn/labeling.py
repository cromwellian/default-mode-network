"""LLM-synthesized cluster labels and per-interest tags.

Given the top-N members closest to a cluster centroid, ask an LLM for a *thematic* label —
not a verbatim title. Falls back to a deterministic word-frequency stub when the LLM is the
no-API stub (so dry-run still produces something usable). The result is a `ClusterLabel`
containing a 3–6 word `theme`, 3–5 short `subtopics`, and a one-sentence `rationale`.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClusterLabel:
    """A synthesized cluster label: thematic name + subtopics + rationale."""

    theme: str
    subtopics: list[str] = field(default_factory=list)
    rationale: str = ""
    members_sampled: int = 0

    def to_meta(self) -> dict:
        """Serialize to the dict shape stored in `clusters.meta`."""
        return {
            "theme": self.theme,
            "subtopics": list(self.subtopics),
            "rationale": self.rationale,
        }


# A small stopword list for the dry-run fallback. Includes the usual function words plus
# domain-noise tokens that appear in URL-flavored interest texts (com/www/youtube/etc.).
_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "at", "is", "are",
    "was", "were", "with", "by", "from", "as", "this", "that", "it", "its", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "but", "not", "no",
    "yes", "you", "your", "yours", "we", "us", "our", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "each", "few", "more", "most", "other", "some", "such", "than", "then",
    "https", "http", "www", "com", "org", "net", "html", "page", "search", "video",
    "videos", "watch", "youtube", "github", "reddit", "google", "gmail", "drive",
    "twitter", "facebook", "linkedin", "chatgpt", "claude", "comments", "subreddit",
    "user", "users", "view", "post", "posts", "thread", "site", "sites", "web",
    "online", "open", "close", "click", "follow", "share", "like", "likes", "more",
    "make", "made", "use", "used", "using", "uses", "see", "seen", "go", "goes",
    "went", "gone", "get", "got", "new",
}


def synthesize_label(
    member_texts: list[str],
    llm,
    rng: Optional[random.Random] = None,
) -> ClusterLabel:
    """Synthesize a thematic cluster label from its top-N nearest-centroid members.

    Falls back to a deterministic stub heuristic if the LLM is the dry-run stub or the
    response can't be parsed as JSON.
    """
    rng = rng or random.Random()
    n = len(member_texts)
    if n == 0:
        return ClusterLabel(theme="misc", rationale="empty cluster", members_sampled=0)
    if getattr(llm, "name", None) == "stub":
        return _stub_label(member_texts)

    items_block = "\n".join(f"{i + 1}. {(t or '')[:200]}" for i, t in enumerate(member_texts))
    user = (
        "You are labeling a cluster of items from a user's taste profile.\n"
        "Below are the items closest to this cluster's center. They share an underlying theme.\n"
        'Return a JSON object: {"theme": "<3-6 word thematic label>", '
        '"subtopics": ["<phrase>", "<phrase>", "<phrase>"], '
        '"rationale": "<one sentence>"}.\n'
        'The theme MUST be conceptual (e.g. "weird math curiosities", "open-source AI tooling", '
        '"contrarian nutrition science"), NOT a verbatim title.\n\n'
        f"Items:\n{items_block}\n\nJSON only."
    )
    try:
        resp = llm.complete(
            system="You are a sharp, concise cluster labeler.",
            user=user,
            max_tokens=300,
        )
        text = resp.text or ""
    except Exception:
        return _stub_label(member_texts)

    parsed = _parse_label_json(text)
    if parsed is not None:
        theme, subtopics, rationale = parsed
        return ClusterLabel(
            theme=theme, subtopics=subtopics, rationale=rationale, members_sampled=n
        )
    m = re.search(r'"theme"\s*:\s*"([^"]+)"', text)
    if m:
        return ClusterLabel(theme=m.group(1)[:80], members_sampled=n)
    return _stub_label(member_texts)


def tag_interest(text: str, taxonomy_hints: list[str], llm) -> list[str]:
    """Return 1-3 short concept tags for an interest, drawn from hints when relevant."""
    if not text:
        return []
    if getattr(llm, "name", None) == "stub":
        return _stub_tags(text, taxonomy_hints)
    hints_block = ", ".join(taxonomy_hints) if taxonomy_hints else "(no hints)"
    user = (
        f"Tag this interest with 1-3 short concept tags (lowercase phrases). "
        f"Prefer drawing from these hints when relevant: {hints_block}. "
        "Return as a comma-separated list, no explanation.\n\n"
        f"Interest: {text[:300]}"
    )
    try:
        resp = llm.complete(system="You tag interests.", user=user, max_tokens=80)
        line = (resp.text or "").strip().splitlines()
    except Exception:
        return _stub_tags(text, taxonomy_hints)
    if not line:
        return _stub_tags(text, taxonomy_hints)
    tags = [t.strip().lower().strip("- ") for t in re.split(r"[,;]", line[0]) if t.strip()]
    return [t for t in tags if t][:3]


def _parse_label_json(text: str) -> Optional[tuple[str, list[str], str]]:
    """Pull a JSON object out of an LLM response (possibly wrapped in code fences) and parse."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    theme = (data.get("theme") or "").strip()
    if not theme:
        return None
    subs_raw = data.get("subtopics") or []
    subs = [s.strip() for s in subs_raw if isinstance(s, str) and s.strip()][:5]
    rationale = (data.get("rationale") or "").strip()
    return theme[:80], subs, rationale[:240]


def _stub_label(member_texts: list[str]) -> ClusterLabel:
    """Deterministic non-LLM label: top-N word-frequency from members. Used in dry-run."""
    words: list[str] = []
    for t in member_texts:
        for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", (t or "").lower()):
            if tok in _STOPWORDS or len(tok) < 4:
                continue
            words.append(tok)
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(8)]
    if not top:
        return ClusterLabel(
            theme="misc", rationale="no salient terms", members_sampled=len(member_texts)
        )
    if len(top) >= 2:
        theme = f"{top[0]} & {top[1]}"
    else:
        theme = top[0]
    subtopics = top[:5]
    rationale = f"Recurring terms: {', '.join(top[:3])}."
    return ClusterLabel(
        theme=theme,
        subtopics=subtopics,
        rationale=rationale,
        members_sampled=len(member_texts),
    )


def _stub_tags(text: str, hints: list[str]) -> list[str]:
    """Stub fallback for tag_interest: pick hints whose tokens appear in text, else first 2 hints."""
    text_lower = (text or "").lower()
    chosen = [h for h in hints if any(w in text_lower for w in h.lower().split() if w)]
    if chosen:
        return chosen[:2]
    if hints:
        return hints[:2]
    words = [
        t
        for t in re.findall(r"[a-zA-Z]{4,}", text_lower)
        if t not in _STOPWORDS
    ]
    if words:
        return [Counter(words).most_common(1)[0][0]]
    return []
