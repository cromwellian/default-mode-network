"""`image_riff` activity (v0.3): turn the seed into a vivid visual prompt + image.

Re-prompts the seed via the LLM into a more cinematic visual brief, then calls the
first available image generator (Nano Banana / Replicate / dry-run stub). Wraps the
result with short LLM commentary on what the image is trying to express.
"""
from __future__ import annotations

from dmn import generators as gens
from dmn.activities import ActivityContext, ActivityResult, register
from dmn.activities._helpers import slugify
from dmn.seeds import Seed

VISUAL_SYSTEM = (
    "You translate research seeds into vivid, concrete one-paragraph visual prompts "
    "for a text-to-image model. Specify subject, composition, mood, palette, lighting, "
    "and any references to art movements or photographers when helpful. Avoid clichés."
)

COMMENTARY_SYSTEM = (
    "You are a curator. In 2-3 sentences, explain what an image generated from a given "
    "visual prompt is trying to express, and why it pairs with the original seed."
)


class ImageRiffActivity:
    """Visual prompt → image (Nano Banana / Replicate / dry-run) → curator commentary."""

    name = "image_riff"
    requires_llm = False  # LLM is optional; without it we use the seed text verbatim
    requires_keys: list[str] = []  # any image backend qualifies; checked dynamically
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        # Available iff *any* image generator can run right now (dry-run-stub or real).
        if ctx.dry_run:
            return any(g for g in gens.available_for("image") if g.name.startswith("dryrun_"))
        real = [g for g in gens.available_for("image") if not g.name.startswith("dryrun_")]
        return len(real) > 0

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        visual_prompt = self._make_visual_prompt(seed, ctx)
        backend = self._pick_backend(ctx)
        if backend is None:
            return ActivityResult(
                title=f"Image (skipped): {seed.text[:60]}",
                body_md=(
                    f"## Image riff\n\n**Seed:** {seed.text}\n\n"
                    "_(no image backend available)_"
                ),
                embedding_text=seed.text,
                metadata={"skipped": True},
            )
        try:
            artifact = backend.generate(visual_prompt)
        except Exception as e:
            return ActivityResult(
                title=f"Image (failed): {seed.text[:60]}",
                body_md=(
                    f"## Image riff\n\n**Seed:** {seed.text}\n\n"
                    f"_(generator `{backend.name}` failed: {e})_"
                ),
                embedding_text=seed.text,
                metadata={"error": str(e), "generator": backend.name},
            )
        commentary = self._curator_blurb(seed, visual_prompt, ctx)
        body = self._render_body(seed, visual_prompt, artifact, commentary, backend.name)
        return ActivityResult(
            title=f"Image: {seed.text[:60]}",
            body_md=body,
            artifacts=[artifact],
            embedding_text=seed.text + " :: " + visual_prompt[:300],
            metadata={
                "generator": backend.name,
                "visual_prompt": visual_prompt,
            },
        )

    def _make_visual_prompt(self, seed: Seed, ctx: ActivityContext) -> str:
        """LLM rewrite of the seed as a one-paragraph visual prompt; falls back to seed."""
        try:
            resp = ctx.llm.complete(
                system=VISUAL_SYSTEM,
                user=f"Seed: {seed.text}\n\nWrite the visual prompt now.",
                max_tokens=200,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        return text or seed.text

    def _curator_blurb(
        self, seed: Seed, visual_prompt: str, ctx: ActivityContext
    ) -> str:
        """Short LLM blurb interpreting the image. Empty string on any LLM failure."""
        try:
            resp = ctx.llm.complete(
                system=COMMENTARY_SYSTEM,
                user=(
                    f"Seed: {seed.text}\n\nVisual prompt: {visual_prompt}\n\n"
                    "Write 2-3 sentences."
                ),
                max_tokens=200,
            )
            return (resp.text or "").strip()
        except Exception:
            return ""

    def _pick_backend(self, ctx: ActivityContext):
        """First available image backend; in dry-run, restrict to the stub."""
        if ctx.dry_run:
            stubs = [g for g in gens.available_for("image") if g.name.startswith("dryrun_")]
            return stubs[0] if stubs else None
        real = [g for g in gens.available_for("image") if not g.name.startswith("dryrun_")]
        return real[0] if real else None

    def _render_body(
        self,
        seed: Seed,
        visual_prompt: str,
        artifact,
        commentary: str,
        generator: str,
    ) -> str:
        """Compose the brief markdown for the journal."""
        lines = [
            f"## Image riff: {seed.text}",
            "",
            f"**Generator:** `{generator}`",
            "",
            "### Visual prompt",
            "",
            f"> {visual_prompt}",
            "",
        ]
        target = artifact.bytes_path or artifact.url or ""
        if target:
            lines.append(f"![generated]({target})")
        if commentary:
            lines.extend(["", "### Curator's note", "", commentary])
        return "\n".join(lines)


register(ImageRiffActivity())
