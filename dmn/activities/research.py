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
from dmn.loop import (
    SYNTHESIS_SYSTEM,
    execute_tools,
    parse_synthesis,
    plan_tools,
)
from dmn.seeds import Seed
from dmn.tools import ResearchItem


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
        if not items:
            body = "_(no tool results)_"
            return ActivityResult(
                title=seed.text,
                body_md=body,
                embedding_text=seed.text,
                metadata={"tools": plan, "items_count": 0},
            )

        # Embed + score each result; keep the top-5 for the synthesis prompt.
        item_embs = ctx.embed_fn(
            [(i.title or "") + " — " + (i.summary or "") for i in items]
        )
        for it, v in zip(items, item_embs):
            it.embedding = v
        scored: list[tuple[float, dict, ResearchItem]] = []
        for it in items:
            d = taste.dopamine(it.embedding, ctx.centroids, ctx.recent_embs, ctx.rng)
            scored.append((d["total"], d, it))
        scored.sort(key=lambda x: x[0], reverse=True)
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
        return ActivityResult(
            title=seed.text,
            body_md=body,
            embedding_text=seed.text + " :: " + body[:1000],
            metadata={
                "tools": plan,
                "items_count": len(items),
                "entities": entities,
                "rabbit_holes": rabbit_holes,
            },
        )


register(ResearchActivity())
