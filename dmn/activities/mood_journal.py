"""`mood_journal` activity (v0.3): a 200-word reflective entry on what your mind seems
drawn to right now. Pure-LLM, no tool calls, atmospheric / diary-flavored.

Pulls the last few high-dopamine briefs from the journal and the user's top cluster
themes, then asks the LLM for a short reflective paragraph. Off by default; opt-in via
`--activities ...mood_journal`.
"""
from __future__ import annotations

from dmn.activities import ActivityContext, ActivityResult, register
from dmn.fulfillment import compute_activity_fulfillment
from dmn.seeds import Seed

SYSTEM = (
    "You write a short, warm, reflective journal entry (~200 words, single paragraph) "
    "about what someone's mind seems drawn to right now. Voice: a thoughtful friend, "
    "not a therapist. Concrete, observational, never grandiose."
)

USER_TEMPLATE = (
    "Their top cluster themes:\n{themes}\n\n"
    "Recent high-dopamine briefs:\n{recent}\n\n"
    "Today's prompt seed: {seed_text}\n\n"
    "Write the reflective entry now (~200 words, one paragraph)."
)


class MoodJournalActivity:
    """Reflective 200-word journal entry pulling from clusters + recent briefs."""

    name = "mood_journal"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        themes = self._summarize_clusters(ctx.clusters[:3])
        recent = self._summarize_recent(ctx)
        try:
            resp = ctx.llm.complete(
                system=SYSTEM,
                user=USER_TEMPLATE.format(
                    themes=themes, recent=recent, seed_text=seed.text
                ),
                max_tokens=400,
            )
            entry = (resp.text or "").strip()
        except Exception:
            entry = ""
        if not entry:
            entry = (
                "Today's wander circles a familiar question with new edges. The themes you've "
                "been pulling at — quietly recurring — keep nudging the same kind of curiosity "
                "from different angles. There's no resolution yet, just a useful texture forming."
            )
        body = "\n".join(
            [
                f"## Mood journal — {seed.text[:60]}",
                "",
                entry,
                "",
                f"_(themes considered: {themes or 'none'})_",
            ]
        )
        fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
            activity=self.name,
            body_md=body,
            artifacts=[],
        )
        return ActivityResult(
            title=f"Mood journal: {seed.text[:60]}",
            body_md=body,
            embedding_text=seed.text + " :: " + entry[:300],
            metadata={
                "themes": themes,
                "recent_count": len(ctx.recent_embs),
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
            },
        )

    def _summarize_clusters(self, clusters: list[dict]) -> str:
        """Compact bullet list of the top clusters (theme + a representative subtopic)."""
        out: list[str] = []
        for c in clusters:
            meta = c.get("meta") or {}
            theme = meta.get("theme") or c.get("label") or "(unlabeled)"
            subs = (meta.get("subtopics") or [])[:2]
            sub_str = ", ".join(subs) if subs else ""
            out.append(f"- {theme}" + (f" — {sub_str}" if sub_str else ""))
        return "\n".join(out)

    def _summarize_recent(self, ctx: ActivityContext) -> str:
        """Pull the last few high-dopamine briefs from SQLite (best-effort, optional)."""
        try:
            from dmn import store  # local import to avoid cycle at module load
            conn = store.connect()
            entries = store.list_journal(conn, limit=20)
            top = sorted(entries, key=lambda e: e.get("dopamine_total") or 0.0, reverse=True)[:3]
            return "\n".join(
                f"- {e.get('seed') or '?'} (d={e.get('dopamine_total') or 0.0:.2f})"
                for e in top
            )
        except Exception:
            return "(none)"


register(MoodJournalActivity())
