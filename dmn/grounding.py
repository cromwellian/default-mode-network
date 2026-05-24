"""Grounding: make the *creation* activities consume external information before they riff.

Mind-wandering isn't pure daydreaming — it's also reading, browsing, and remixing what
others have made. Only the `research` activity hit external sources before this module;
`code_sketch` / `web_app_sketch` / `app_idea` / `algorithm_explore` / `ml_experiment`
generated from the bare seed (pure LLM hallucination). `gather_grounding` runs a quick
sweep of reliable no-auth sources on the seed's clean query and returns real references;
activities inject them into their prompt ("build on these, don't invent from scratch") and
into the reward (a grounded brief out-scores an ungrounded one).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from dmn.loop import available_tools_for, execute_tools
from dmn.seeds import Seed, search_query
from dmn.tools import ResearchItem

if TYPE_CHECKING:  # avoid a circular import at runtime
    from dmn.activities import ActivityContext

# Preferred grounding sources, in priority order. We use a fixed sweep rather than the
# LLM tool-planner: grounding wants breadth (what's out there) and skipping the planner
# call keeps the extra cost to the searches themselves.
_PREFERRED = ["wikipedia", "arxiv", "semantic_scholar", "websearch", "hackernews", "reddit"]


def gather_grounding(
    seed: Seed, ctx: "ActivityContext", *, max_items: int = 6, max_tools: int = 4
) -> list[ResearchItem]:
    """Search reliable sources for real references related to `seed`. Cached per query.

    Returns an empty list when grounding is disabled (`ctx.ground is False`), when no
    tools are available, or when every search whiffs. Never raises.
    """
    if not getattr(ctx, "ground", True):
        return []
    query = (seed.query or "").strip() or search_query(seed.text, subtopic=seed.subtopic)
    if not query:
        return []
    cache = getattr(ctx, "grounding_cache", None)
    key = query.lower()
    if cache is not None and key in cache:
        return cache[key]

    avail = available_tools_for(ctx.dry_run)
    tools = [t for t in _PREFERRED if t in avail][:max_tools] or avail[:3]
    try:
        items = execute_tools(tools, query, verbose=ctx.verbose)
    except Exception:
        items = []

    seen: set[tuple[str, str]] = set()
    out: list[ResearchItem] = []
    for it in items:
        k = ((it.title or "").strip().lower(), (it.url or "").strip())
        if k in seen or not (it.title or "").strip():
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= max_items:
            break
    if cache is not None:
        cache[key] = out
    return out


def format_grounding(items: list[ResearchItem], *, limit: int = 6) -> str:
    """Render references as a compact bullet list for prompt injection."""
    lines: list[str] = []
    for it in items[:limit]:
        summary = " ".join((it.summary or "").split())[:180]
        lines.append(f"- [{it.source}] {it.title}: {summary} <{it.url}>")
    return "\n".join(lines)


def grounding_block(items: list[ResearchItem]) -> str:
    """The instruction + references block injected into a creation activity's prompt."""
    refs = format_grounding(items)
    if not refs:
        return (
            "No external references turned up for this, so proceed from first principles — "
            "but stay concrete about how it actually works; don't hand-wave."
        )
    return (
        "Real things others have published on this. Ground your work in them: build on, "
        "extend, remix, or critique these rather than inventing from scratch, and say where "
        "you diverge.\n" + refs
    )


def grounding_footer(items: list[ResearchItem], *, limit: int = 6) -> str:
    """A markdown 'Grounded in' footer appended to a grounded brief's body."""
    if not items:
        return ""
    lines = ["", "---", "**Grounded in:**"]
    for it in items[:limit]:
        title = it.title or it.url or "(source)"
        if it.url:
            lines.append(f"- [{title}]({it.url}) — {it.source}")
        else:
            lines.append(f"- {title} — {it.source}")
    return "\n".join(lines)


def grounding_fulfillment(items: list[ResearchItem]) -> float:
    """Reward signal: grounded creation beats ungrounded daydreaming.

    Two-plus references → full credit; one → slight discount; none → a gentle penalty so a
    brief that consumed nothing scores below one that did.
    """
    n = len(items)
    if n >= 2:
        return 1.0
    if n == 1:
        return 0.85
    return 0.65
