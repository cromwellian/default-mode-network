"""`video_riff` activity (v0.3): turn the seed into a short cinematic clip.

Available iff a real video backend is configured. High latency makes this a poor
default for routine wanders, so the activity defaults to off in any sensible mix.
"""
from __future__ import annotations

import os

from dmn import generators as gens
from dmn.activities import ActivityContext, ActivityResult, register
from dmn.fulfillment import compute_activity_fulfillment
from dmn.journal import JOURNAL_DIR
from dmn.paths import artifact_href_for_journal
from dmn.activities.riff_prompts import make_video_prompt
from dmn.seeds import Seed


class VideoRiffActivity:
    """Cinematic prompt → video (Runway / Replicate / dry-run stub)."""

    name = "video_riff"
    requires_llm = False
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        if ctx.dry_run:
            return any(g for g in gens.available_for("video") if g.name.startswith("dryrun_"))
        if os.environ.get("DMN_ENABLE_VIDEO_RIFFS", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return False
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
        artifact = None
        errors: list[str] = []
        for backend in self._backends(ctx):
            try:
                artifact = backend.generate(prompt)
                break
            except Exception as e:
                errors.append(f"{backend.name}: {e}")
        if artifact is None:
            err = "; ".join(errors) if errors else "no video backend available"
            body = (
                f"## Video riff\n\n**Seed:** {seed.text}\n\n"
                f"_(all generators failed: {err})_"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"Video (failed): {seed.text[:60]}",
                body_md=body,
                embedding_text=seed.text,
                metadata={
                    "error": err,
                    "generator": backend.name if backend else None,
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
                f"**Generator:** `{artifact.generator}`",
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
                "generator": artifact.generator,
                "video_prompt": prompt,
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
            },
        )

    def _make_prompt(self, seed: Seed, ctx: ActivityContext) -> str:
        return make_video_prompt(seed, ctx)

    def _pick_backend(self, ctx: ActivityContext):
        backends = self._backends(ctx)
        return backends[0] if backends else None

    def _backends(self, ctx: ActivityContext):
        if ctx.dry_run:
            return [g for g in gens.available_for("video") if g.name.startswith("dryrun_")]
        real = [g for g in gens.available_for("video") if not g.name.startswith("dryrun_")]
        preferred = [g for g in real if g.name == "runway_video"]
        return preferred + [g for g in real if g.name != "runway_video"]


register(VideoRiffActivity())
