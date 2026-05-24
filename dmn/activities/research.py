"""`research` activity (v0.3): the v0.2.x synthesis flow as a first-class activity.

Lifts the search → score → synthesize flow that used to live inside `explore.py` /
`wander.py` directly. With activities in place, "research" is just one option in the
mix — but it's the default, so v0.2.1 behavior is preserved verbatim when no other
activities are requested.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from dmn import taste
from dmn.activities import ActivityContext, ActivityResult, register
from dmn.fulfillment import compute_fulfillment
from dmn.loop import (
    SYNTHESIS_SYSTEM,
    available_tools_for,
    execute_tools,
    parse_synthesis,
    plan_tools,
)
from dmn.seeds import Seed
from dmn.tools import ResearchItem

SKIP_FULFILLMENT_THRESHOLD = 0.15
RETRY_FULFILLMENT_THRESHOLD = 0.25


def _pick_retry_tool(plan: list[str], rng: random.Random, dry_run: bool) -> Optional[str]:
    """Pick an available tool not used in the first plan."""
    avail = available_tools_for(dry_run)
    unused = [t for t in avail if t not in plan]
    if not unused:
        return None
    return rng.choice(unused)


def _score_items(
    items: list[ResearchItem],
    ctx: ActivityContext,
) -> tuple[list[tuple[float, dict, ResearchItem]], list[float], list[float]]:
    item_embs = ctx.embed_fn(
        [(i.title or "") + " — " + (i.summary or "") for i in items]
    )
    for it, v in zip(items, item_embs):
        it.embedding = v
    scored: list[tuple[float, dict, ResearchItem]] = []
    item_totals: list[float] = []
    alignments: list[float] = []
    for it in items:
        d = taste.dopamine(it.embedding, ctx.centroids, ctx.recent_embs, ctx.rng)
        a = taste.alignment(it.embedding, ctx.centroids)
        scored.append((d["total"], d, it))
        item_totals.append(d["total"])
        alignments.append(a)
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored, item_totals, alignments


class ResearchActivity:
    """The classic wander activity: search → score → synthesize → brief."""

    name = "research"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        # Research always works: stub LLM + a no-auth tool (Wikipedia) suffice in dry-run.
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        plan = plan_tools(seed.text, ctx.llm, ctx.dry_run, ctx.rng)
        items = execute_tools(plan, seed.text, verbose=ctx.verbose)
        scored: list[tuple[float, dict, ResearchItem]] = []
        item_totals: list[float] = []
        alignments: list[float] = []

        if items:
            scored, item_totals, alignments = _score_items(items, ctx)

        fulfillment, breakdown = compute_fulfillment(
            tool_items=items,
            item_scores=item_totals,
            item_alignments=alignments,
            activity="research",
        )

        if fulfillment < SKIP_FULFILLMENT_THRESHOLD and len(items) == 0:
            if ctx.verbose:
                print(
                    f"[yellow]  research skipped: fulfillment={fulfillment:.2f}, "
                    f"no tool results[/]"
                )
            return ActivityResult(
                title=seed.text,
                body_md="",
                embedding_text=seed.text,
                metadata={
                    "skipped": True,
                    "reason": "empty_search",
                    "tools": plan,
                    "items_count": 0,
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": breakdown,
                },
            )

        if fulfillment < RETRY_FULFILLMENT_THRESHOLD and items:
            retry_tool = _pick_retry_tool(plan, ctx.rng, ctx.dry_run)
            if retry_tool:
                if ctx.verbose:
                    print(
                        f"[yellow]  low fulfillment ({fulfillment:.2f}); "
                        f"retrying with {retry_tool}[/]"
                    )
                retry_items = execute_tools([retry_tool], seed.text, verbose=ctx.verbose)
                if retry_items:
                    retry_scored, retry_totals, retry_alignments = _score_items(
                        retry_items, ctx
                    )
                    retry_f, retry_breakdown = compute_fulfillment(
                        tool_items=retry_items,
                        item_scores=retry_totals,
                        item_alignments=retry_alignments,
                        activity="research",
                    )
                    if retry_f >= fulfillment:
                        items = retry_items
                        scored = retry_scored
                        item_totals = retry_totals
                        alignments = retry_alignments
                        fulfillment = retry_f
                        breakdown = retry_breakdown
                        plan = plan + [retry_tool]

        if not items:
            return ActivityResult(
                title=seed.text,
                body_md="",
                embedding_text=seed.text,
                metadata={
                    "skipped": True,
                    "reason": "empty_search_after_retry",
                    "tools": plan,
                    "items_count": 0,
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": breakdown,
                },
            )

        top_k = scored[:5]
        bullets = [
            f"- [{it.source}] **{it.title}** — {it.summary[:240]}\n"
            f"  {it.url}\n"
            f"  dopamine: {d}"
            for _, d, it in top_k
        ]
        gathered = "\n".join(bullets)
        prompt = (
            f"Seed question: {seed.text}\n\n"
            f"Gathered findings:\n{gathered}\n\n"
            "Write the brief now."
        )
        try:
            resp = ctx.llm.complete(
                system=SYNTHESIS_SYSTEM, user=prompt, max_tokens=800
            )
            raw_body = (resp.text or "").strip()
        except Exception:
            raw_body = f"_(synthesis failed; raw findings)_\n\n{gathered}"
        body, entities, rabbit_holes = parse_synthesis(raw_body)

        fulfillment, breakdown = compute_fulfillment(
            tool_items=items,
            item_scores=item_totals,
            item_alignments=alignments,
            body_md=body,
            activity="research",
        )

        return ActivityResult(
            title=seed.text,
            body_md=body,
            embedding_text=seed.text + " :: " + body[:1000],
            metadata={
                "tools": plan,
                "items_count": len(items),
                "entities": entities,
                "rabbit_holes": rabbit_holes,
                "fulfillment": fulfillment,
                "fulfillment_breakdown": breakdown,
            },
        )


register(ResearchActivity())
