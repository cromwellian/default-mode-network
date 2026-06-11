"""Shared exploration helpers used by both `explore.py` (flat mode) and `wander.py` (tree mode).

Factored out in v0.2 so wander mode reuses the same tool-planning + tool-execution +
synthesis-prompt as flat mode. Behavior is byte-identical to v0.1.x for flat callers.

v0.2.1: the synthesis prompt now asks the LLM to append a fenced ```json block with
`entities` and `rabbit_holes`. `parse_synthesis` extracts that block (regex + json.loads,
robust to absence) and returns (clean_body, entities, rabbit_holes). Stub-LLM dry-runs
won't produce the JSON block; the parser yields empty lists in that case.
"""
from __future__ import annotations

import json
import random
import re
import signal
from typing import Callable, Optional

from dmn.tools import ResearchItem
from dmn.tools import available as tools_available
from dmn.tools import get as get_tool

# Tools that work without API keys; used when --dry-run is set.
SAFE_TOOLS = ["wikipedia", "websearch", "hackernews"]


def install_graceful_sigint(console=None) -> dict:
    """First Ctrl-C requests a graceful stop (loops check the returned flag and finish
    the current iteration, so SQLite and the journal stay consistent); a second Ctrl-C
    raises KeyboardInterrupt as usual. Returns {"stop": bool} for loops to poll."""
    state = {"stop": False}

    def _handler(signum, frame):
        if state["stop"]:
            raise KeyboardInterrupt
        state["stop"] = True
        if console is not None:
            console.print(
                "\n[yellow]Finishing this iteration, then stopping — progress will be "
                "saved. (Ctrl-C again to force quit.)[/]"
            )

    signal.signal(signal.SIGINT, _handler)
    return state

SYNTHESIS_SYSTEM = (
    "You are the inner voice of a wandering mind. Synthesize the gathered findings into a "
    "short markdown brief (150-300 words) for the user. Voice: a smart friend texting them "
    "something cool they just found. Always include three sections (use headings): "
    "1) **What surprised me**, "
    "2) **One thing you'll find delightful**, "
    "3) **A rabbit hole for tomorrow**. "
    "Cite sources inline as [title](url) when relevant.\n\n"
    "AFTER the markdown brief, append a single fenced ```json block (and nothing after it) "
    "containing two keys:\n"
    "- `entities`: list of objects with `name` (str), `type` (one of: person | org | paper "
    "| concept | topic | place | tool), and `salience` (float 0..1). Pick 3-7 of the most "
    "concrete, specific entities the brief actually leans on.\n"
    "- `rabbit_holes`: list of 2-4 short strings naming side-threads worth pursuing.\n"
    "Example tail (one example only — do not include this verbatim):\n"
    "```json\n"
    '{"entities": [{"name": "Constitutional AI", "type": "concept", "salience": 0.9}], '
    '"rabbit_holes": ["annotator disagreement modeling"]}\n'
    "```"
)


# Capture any fenced ```json ... ``` block (greedy across newlines). Used to extract
# the structured tail that the v0.2.1 synthesis prompt asks for. We deliberately match
# only `json`-typed fences so plain code samples in the brief body are left intact.
_JSON_FENCE_RE = re.compile(
    r"```\s*json\s*\n(.*?)\n\s*```", re.DOTALL | re.IGNORECASE
)


def parse_synthesis(raw_text: str) -> tuple[str, list[dict], list[str]]:
    """Split a raw synthesis response into (clean_markdown, entities, rabbit_holes).

    Strategy:
      1. Find the LAST ```json fenced block (the prompt asks for it at the tail; a real
         LLM occasionally hallucinates an example earlier, so we take the last one).
      2. Try `json.loads` on its body. On any failure, return empty lists and leave the
         body untouched.
      3. On success, strip the fenced block from the body so the markdown stays clean.

    Stub-LLM dry-runs don't produce the JSON block and just round-trip ([], []).
    """
    raw_text = raw_text or ""
    matches = list(_JSON_FENCE_RE.finditer(raw_text))
    if not matches:
        return raw_text.strip(), [], []
    last = matches[-1]
    payload = last.group(1).strip()
    try:
        data = json.loads(payload)
    except Exception:
        return raw_text.strip(), [], []
    if not isinstance(data, dict):
        return raw_text.strip(), [], []
    raw_entities = data.get("entities") or []
    raw_rabbits = data.get("rabbit_holes") or []
    entities: list[dict] = []
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        if not name:
            continue
        ent_type = (e.get("type") or "concept").strip().lower()
        try:
            salience = float(e.get("salience", 0.5))
        except (TypeError, ValueError):
            salience = 0.5
        salience = max(0.0, min(1.0, salience))
        entities.append({"name": name, "type": ent_type, "salience": salience})
    rabbit_holes: list[str] = []
    for rh in raw_rabbits:
        if isinstance(rh, str):
            rh = rh.strip()
            if rh:
                rabbit_holes.append(rh)
    # Strip the trailing fenced block from the markdown body.
    clean_body = (raw_text[: last.start()] + raw_text[last.end():]).strip()
    return clean_body, entities, rabbit_holes


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
