"""`video_riff` activity (v0.3, stub-grade): turn the seed into a short cinematic clip.

Available iff `REPLICATE_API_TOKEN` is set (via `dmn.generators.video_replicate`). High
latency makes this a poor default for routine wanders, so the activity defaults to off
in any sensible mix.
"""
from __future__ import annotations

from dmn import generators as gens
from dmn.activities import ActivityContext, ActivityResult, register
from dmn.fulfillment import compute_activity_fulfillment
from dmn.journal import JOURNAL_DIR
from dmn.paths import artifact_href_for_journal
from dmn.seeds import Seed

VIDEO_SYSTEM = (
    "You translate research seeds into one-paragraph cinematic-direction prompts for a "
    "text-to-video model. Specify subject, camera move, lighting, mood, and pacing."
)


class VideoRiffActivity:
    """Cinematic prompt → video (Replicate / dry-run stub)."""

    name = "video_riff"
    requires_llm = False
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        if ctx.dry_run:
            return any(g for g in gens.available_for("video") if g.name.startswith("dryrun_"))
        real = [g for g in gens.available_for("video") if not g.name.startswith("dryrun_")]
        return len(real) > 0

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        prompt = self._make_prompt(seed, ctx)
        backend = self._pick_backend(ctx)
        if backend is None:
            body = (
                f"## Video riff\n\n**Seed:** {seed.text}\n\n"
                "_(no video backend available)_"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"Video (skipped): {seed.text[:60]}",
                body_md=body,
                embedding_text=seed.text,
                metadata={
                    "skipped": True,
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": fulfillment_breakdown,
                },
            )
        try:
            artifact = backend.generate(prompt)
        except Exception as e:
            body = (
                f"## Video riff\n\n**Seed:** {seed.text}\n\n"
                f"_(generator `{backend.name}` failed: {e})_"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"Video (failed): {seed.text[:60]}",
                body_md=body,
                embedding_text=seed.text,
                metadata={
                    "error": str(e),
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": fulfillment_breakdown,
                },
            )
        raw = artifact.bytes_path or artifact.url or ""
        target = (
            artifact_href_for_journal(JOURNAL_DIR, raw) if raw else ""
        )
        body = "\n".join(
            [
                f"## Video riff: {seed.text}",
                "",
                f"**Generator:** `{backend.name}`",
                "",
                "### Cinematic direction",
                "",
                f"> {prompt}",
                "",
                f"[Watch]({target})" if target else "_(no output bytes)_",
            ]
        )
        fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
            activity=self.name,
            body_md=body,
            artifacts=[artifact],
        )
        return ActivityResult(
            title=f"Video: {seed.text[:60]}",
            body_md=body,
            artifacts=[artifact],
            embedding_text=seed.text + " :: " + prompt[:300],
            metadata={
                "generator": backend.name,
                "video_prompt": prompt,
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
            },
        )

    def _make_prompt(self, seed: Seed, ctx: ActivityContext) -> str:
        try:
            resp = ctx.llm.complete(
                system=VIDEO_SYSTEM,
                user=f"Seed: {seed.text}\n\nWrite the cinematic direction now.",
                max_tokens=200,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        return text or seed.text

    def _pick_backend(self, ctx: ActivityContext):
        if ctx.dry_run:
            stubs = [g for g in gens.available_for("video") if g.name.startswith("dryrun_")]
            return stubs[0] if stubs else None
        real = [g for g in gens.available_for("video") if not g.name.startswith("dryrun_")]
        return real[0] if real else None


register(VideoRiffActivity())
