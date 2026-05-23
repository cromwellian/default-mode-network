"""Shared exploration helpers used by both `explore.py` (flat mode) and `wander.py` (tree mode).

Factored out in v0.2 so wander mode reuses the same tool-planning + tool-execution +
synthesis-prompt as flat mode. Behavior is byte-identical to v0.1.x for flat callers.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from dmn.tools import ResearchItem
from dmn.tools import available as tools_available
from dmn.tools import get as get_tool

# Tools that work without API keys; used when --dry-run is set.
SAFE_TOOLS = ["wikipedia", "websearch", "hackernews"]

SYNTHESIS_SYSTEM = (
    "You are the inner voice of a wandering mind. Synthesize the gathered findings into a "
    "short markdown brief (150-300 words) for the user. Voice: a smart friend texting them "
    "something cool they just found. Always include three sections (use headings): "
    "1) **What surprised me**, "
    "2) **One thing you'll find delightful**, "
    "3) **A rabbit hole for tomorrow**. "
    "Cite sources inline as [title](url) when relevant."
)


def plan_tools(
    seed_text: str, llm, dry_run: bool, rng: random.Random
) -> list[str]:
    """Pick 1-3 tools to use for a seed.

    In dry-run, anchor on Wikipedia + one random no-auth tool. (Wikipedia is by far the
    most reliable no-auth tool for natural-language seeds; HN's Algolia search is
    exact-phrase and DDG's package surface is in flux, so always include Wikipedia so
    dry-run produces non-empty results.)
    """
    avail = tools_available()
    if dry_run:
        avail = [t for t in avail if t in SAFE_TOOLS]
    if not avail:
        return []
    if dry_run:
        plan: list[str] = []
        if "wikipedia" in avail:
            plan.append("wikipedia")
        rest = [t for t in avail if t not in plan]
        if rest:
            plan.append(rng.choice(rest))
        return plan or rng.sample(avail, k=min(2, len(avail)))
    prompt = (
        f"Question: {seed_text}\n"
        f"Available tools: {', '.join(avail)}\n"
        "Pick the 1-3 most useful tools, comma-separated, no explanation."
    )
    try:
        resp = llm.complete(
            system="You are a research planner.", user=prompt, max_tokens=80
        )
        chosen = [
            t.strip() for t in (resp.text or "").split(",") if t.strip() in avail
        ]
    except Exception:
        chosen = []
    return chosen[:3] or rng.sample(avail, k=min(2, len(avail)))


def execute_tools(
    tool_names: list[str],
    query: str,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> list[ResearchItem]:
    """Run each chosen tool's `search(query)` and merge results, swallowing per-tool errors."""
    items: list[ResearchItem] = []
    for name in tool_names:
        fn = get_tool(name)
        if not fn:
            continue
        try:
            sub = fn(query) or []
        except Exception as e:
            if verbose and log is not None:
                try:
                    log(f"[yellow]  {name}: {e}[/]")
                except Exception:
                    pass
            continue
        items.extend(sub[:5])
    return items


def available_tools_for(dry_run: bool) -> list[str]:
    """The list of tools the wander/explore loop is willing to call (filtered by dry-run)."""
    avail = tools_available()
    if dry_run:
        avail = [t for t in avail if t in SAFE_TOOLS]
    return avail
