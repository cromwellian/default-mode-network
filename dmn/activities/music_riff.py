"""`music_riff` activity (v0.3): turn the seed into a musical-direction prompt + audio.

Re-prompts the seed via the LLM into a brief musical-direction description (genre,
instruments, tempo, mood), then calls the first available music generator. Falls back
to the dry-run stub when no real backend is configured.
"""
from __future__ import annotations

from dmn import generators as gens
from dmn.activities import ActivityContext, ActivityResult, register
from dmn.fulfillment import compute_activity_fulfillment
from dmn.journal import JOURNAL_DIR
from dmn.paths import artifact_href_for_journal
from dmn.seeds import Seed

MUSIC_SYSTEM = (
    "You translate research seeds into one-paragraph musical-direction prompts for a "
    "text-to-audio model. Specify genre, instrumentation, tempo (bpm), mood, and any "
    "structural notes (intro/build/release). Be concrete and brief."
)


class MusicRiffActivity:
    """Musical-direction prompt → audio (Stable Audio / Lyria / dry-run) → embedded brief."""

    name = "music_riff"
    requires_llm = False
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        if ctx.dry_run:
            return any(g for g in gens.available_for("music") if g.name.startswith("dryrun_"))
        real = [g for g in gens.available_for("music") if not g.name.startswith("dryrun_")]
        return len(real) > 0

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        music_prompt = self._make_prompt(seed, ctx)
        backend = self._pick_backend(ctx)
        if backend is None:
            body = (
                f"## Music riff\n\n**Seed:** {seed.text}\n\n"
                "_(no music backend available)_"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"Music (skipped): {seed.text[:60]}",
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
                artifact = backend.generate(music_prompt)
                break
            except Exception as e:
                errors.append(f"{backend.name}: {e}")
        if artifact is None:
            err = "; ".join(errors) if errors else "no music backend available"
            body = (
                f"## Music riff\n\n**Seed:** {seed.text}\n\n"
                f"_(all generators failed: {err})_"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"Music (failed): {seed.text[:60]}",
                body_md=body,
                embedding_text=seed.text,
                metadata={
                    "error": err,
                    "generator": backend.name if backend else None,
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": fulfillment_breakdown,
                },
            )
        body = self._render_body(seed, music_prompt, artifact, artifact.generator)
        fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
            activity=self.name,
            body_md=body,
            artifacts=[artifact],
        )
        return ActivityResult(
            title=f"Music: {seed.text[:60]}",
            body_md=body,
            artifacts=[artifact],
            embedding_text=seed.text + " :: " + music_prompt[:300],
            metadata={
                "generator": artifact.generator,
                "music_prompt": music_prompt,
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
            },
        )

    def _make_prompt(self, seed: Seed, ctx: ActivityContext) -> str:
        try:
            resp = ctx.llm.complete(
                system=MUSIC_SYSTEM,
                user=f"Seed: {seed.text}\n\nWrite the musical direction now.",
                max_tokens=200,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        return text or seed.text

    def _backends(self, ctx: ActivityContext):
        """Available music backends in priority order; dry-run uses the stub only."""
        if ctx.dry_run:
            return [g for g in gens.available_for("music") if g.name.startswith("dryrun_")]
        return [g for g in gens.available_for("music") if not g.name.startswith("dryrun_")]

    def _pick_backend(self, ctx: ActivityContext):
        backends = self._backends(ctx)
        return backends[0] if backends else None

    def _render_body(
        self, seed: Seed, music_prompt: str, artifact, generator: str
    ) -> str:
        raw = artifact.bytes_path or artifact.url or ""
        seconds = artifact.seconds or "?"
        lines = [
            f"## Music riff: {seed.text}",
            "",
            f"**Generator:** `{generator}` · ~{seconds}s",
            "",
            "### Musical direction",
            "",
            f"> {music_prompt}",
            "",
        ]
        if raw:
            target = artifact_href_for_journal(JOURNAL_DIR, raw)
            lines.extend(
                [
                    f'<audio controls src="{target}"></audio>',
                    "",
                    f"[Listen]({target})",
                ]
            )
        return "\n".join(lines)


register(MusicRiffActivity())
