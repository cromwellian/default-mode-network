"""Subject-aware prompt builders for image/music/video riffs.

Riffs should *illustrate* the wander's subject — explanatory diagrams for technical
topics, recognizable motifs for artistic ones — not produce unrelated surreal imagery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dmn.grounding import format_grounding, gather_grounding
from dmn.seeds import Seed

if TYPE_CHECKING:
    from dmn.activities import ActivityContext

_TECH_CUES = [
    "algorithm", "api", "architecture", "system", "model", "data", "network",
    "protocol", "compression", "neural", "transformer", "database", "compiler",
    "distributed", "quantum", "biology", "chemistry", "physics", "math",
    "statistics", "engineering", "mechanism", "process", "flow", "diagram",
    "structure", "graph", "topology", "embedding", "gradient", "inference",
    "hardware", "software", "research", "science", "technical", "explain",
]

_ART_CUES = [
    "art", "artist", "painting", "music", "jazz", "aesthetic", "film", "cinema", "poetry",
    "literature", "novel", "album", "song", "dance", "theater", "design", "craft",
    "visual culture", "photography", "sculpture", "illustration", "creative",
]


@dataclass
class SubjectContext:
    """Everything a riff prompt needs besides the raw seed."""

    seed_text: str
    cluster_label: str
    brief_excerpt: str
    grounding_excerpt: str
    mode: str  # technical | artistic | narrative | general
    subject_summary: str


def classify_subject(
    seed_text: str,
    brief_md: str = "",
    cluster_label: str = "",
) -> str:
    """Heuristic subject mode for prompt strategy selection."""
    text = f"{seed_text} {brief_md} {cluster_label}".lower()
    tech = sum(1 for c in _TECH_CUES if c in text)
    art = sum(1 for c in _ART_CUES if c in text)
    if tech >= 2 and tech >= art:
        return "technical"
    if art >= 2 and art > tech:
        return "artistic"
    if brief_md and len(brief_md.split()) > 40:
        return "narrative"
    return "general"


def _read_journal_body(path_str: str) -> str:
    if not path_str:
        return ""
    from pathlib import Path

    p = Path(path_str)
    if not p.is_file():
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return raw.strip()


def _first_paragraph(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#+\s", line):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = chunks[0] if chunks else text
    return out[:limit].strip()


def _summarize_subject(seed: Seed, ctx: "ActivityContext", brief_md: str) -> str:
    """One-line subject anchor for the generator prompt."""
    parts = [seed.text.strip()]
    if seed.subtopic:
        parts.append(f"subtopic: {seed.subtopic}")
    if ctx.cluster_label:
        parts.append(f"theme: {ctx.cluster_label}")
    excerpt = _first_paragraph(brief_md, 220)
    if excerpt:
        parts.append(f"brief: {excerpt}")
    return " · ".join(parts)


def build_subject_context(seed: Seed, ctx: "ActivityContext") -> SubjectContext:
    """Assemble subject matter from seed, cluster, parent brief, and optional grounding."""
    brief_md = (getattr(ctx, "parent_brief_md", None) or "").strip()
    cluster = (getattr(ctx, "cluster_label", None) or "").strip()

    grounding_excerpt = ""
    if not brief_md and getattr(ctx, "ground", True):
        try:
            items = gather_grounding(seed, ctx, max_items=4, max_tools=3)
            grounding_excerpt = format_grounding(items, limit=4)
        except Exception:
            grounding_excerpt = ""

    mode = classify_subject(seed.text, brief_md, cluster)
    summary = _summarize_subject(seed, ctx, brief_md)
    brief_excerpt = _first_paragraph(brief_md, 800) if brief_md else ""

    return SubjectContext(
        seed_text=seed.text,
        cluster_label=cluster,
        brief_excerpt=brief_excerpt,
        grounding_excerpt=grounding_excerpt,
        mode=mode,
        subject_summary=summary,
    )


def load_parent_brief_md(parent_row: Optional[dict]) -> str:
    """Read a parent journal row's markdown body when expanding a riff child."""
    if not parent_row:
        return ""
    return _read_journal_body(str(parent_row.get("path") or ""))


def _user_block(subject: SubjectContext, modality: str) -> str:
    lines = [
        f"Seed question: {subject.seed_text}",
    ]
    if subject.cluster_label:
        lines.append(f"Taste cluster theme: {subject.cluster_label}")
    if subject.brief_excerpt:
        lines.append(
            "What the wander brief already established (illustrate THIS, do not drift away):\n"
            f"{subject.brief_excerpt}"
        )
    if subject.grounding_excerpt:
        lines.append(
            "Reference facts to reflect visually/aurally when relevant:\n"
            f"{subject.grounding_excerpt}"
        )
    lines.append(
        f"\nWrite the {modality} generation prompt now — one dense paragraph, "
        "concrete nouns, no meta-commentary."
    )
    return "\n\n".join(lines)


_IMAGE_SYSTEM = {
    "technical": (
        "You write text-to-image prompts for clear, beautiful EXPLANATORY diagrams. "
        "Depict the specific concept from the brief — flowcharts, labeled cross-sections, "
        "comparison panels, system architecture sketches, or infographic layouts. "
        "Include arrows and relationships. Use minimal, exact, simple labels only where "
        "they carry information; prefer short terms from the prompt and avoid dense prose "
        "because image models often misspell long text. Readable and informative; "
        "NOT surreal, NOT unrelated dreamscapes, NOT generic stock photo vibes. "
        "A viewer should immediately know what topic is being explained."
    ),
    "artistic": (
        "You write text-to-image prompts where the scene unmistakably evokes the subject. "
        "Choose motifs, settings, objects, palette, and composition tied to the topic so "
        "viewers say 'that fits the subject'. Visually striking but on-topic — avoid surreal "
        "detours unrelated to the brief."
    ),
    "narrative": (
        "You write text-to-image prompts that capture the story or argument of the brief. "
        "Show a scene or tableau that embodies the key idea, with specific details from the "
        "text — not abstract wallpaper."
    ),
    "general": (
        "You write text-to-image prompts that illustrate a specific research seed. "
        "Prefer explanatory clarity (diagram, labeled scene, metaphor with recognizable "
        "objects) over surreal abstraction. The image must connect to the subject matter."
    ),
}

_MUSIC_SYSTEM = {
    "technical": (
        "You write text-to-audio prompts for music/sound that evokes a technical or scientific "
        "subject WITHOUT lyrics unless requested. Specify instrumentation, rhythm, tempo, "
        "texture, and mood that metaphorically fit the concept (e.g. ticking precision for "
        "algorithms, layered pulses for networks). Listeners should sense the topic."
    ),
    "artistic": (
        "You write musical-direction prompts tightly aligned with the subject's cultural or "
        "emotional world — genre, instruments, tempo, mood, structure. The piece should "
        "feel like it belongs to the topic, not random ambient noise."
    ),
    "narrative": (
        "You write musical-direction prompts that soundtrack the narrative arc of the brief — "
        "motifs, tension, release, instrumentation that follow the story's beats."
    ),
    "general": (
        "You write musical-direction prompts anchored to a specific seed question. "
        "Concrete genre, instrumentation, tempo (bpm), mood, and structure — on-topic."
    ),
}

_VIDEO_SYSTEM = {
    "technical": (
        "You write text-to-video prompts for short explanatory clips: slow pans across "
        "diagrams, animated flow of a process, comparison of before/after, documentary "
        "b-roll that teaches the concept. Clear, legible, on-subject — not surreal."
    ),
    "artistic": (
        "You write cinematic prompts where every shot element reinforces the subject — "
        "setting, props, lighting, camera move. Evocative but recognizable."
    ),
    "narrative": (
        "You write cinematic prompts that dramatize the brief's core idea in 5–10 seconds "
        "of screen time — one clear visual story beat."
    ),
    "general": (
        "You write text-to-video prompts illustrating a research seed with specific visual "
        "details, camera move, lighting, and pacing — on-topic, not abstract."
    ),
}


def _fallback_image(subject: SubjectContext) -> str:
    anchor = subject.brief_excerpt or subject.seed_text
    if subject.mode == "technical":
        return (
            f"Clean educational infographic diagram explaining: {anchor}. "
            "Arrows showing relationships, minimal exact labels, no paragraphs of rendered text, "
            "white or dark neutral background, "
            "readable typography, scientific illustration style, highly specific, not surreal."
        )
    return (
        f"Illustration clearly depicting the subject: {anchor}. "
        "Recognizable objects and setting tied to the topic, cohesive composition, "
        "on-theme color palette, not abstract unrelated surrealism."
    )


def _fallback_music(subject: SubjectContext) -> str:
    anchor = subject.brief_excerpt or subject.seed_text
    return (
        f"Instrumental piece evoking: {anchor}. "
        "Specific genre and instrumentation, moderate tempo, mood aligned with the subject, "
        "clear melodic motif, 30 seconds, no random ambient drift."
    )


def _fallback_video(subject: SubjectContext) -> str:
    anchor = subject.brief_excerpt or subject.seed_text
    if subject.mode == "technical":
        return (
            f"Short explainer clip: animated diagram of {anchor}. "
            "Slow camera push-in, clean labels, documentary lighting, educational pacing."
        )
    return (
        f"Cinematic 10-second shot illustrating: {anchor}. "
        "Specific setting and subject matter visible, steady camera, on-topic, not surreal."
    )


def _llm_prompt(
    ctx: "ActivityContext",
    subject: SubjectContext,
    systems: dict[str, str],
    modality_label: str,
    fallback_fn,
    max_tokens: int = 280,
) -> str:
    system = systems.get(subject.mode) or systems["general"]
    user = _user_block(subject, modality_label)
    try:
        resp = ctx.llm.complete(system=system, user=user, max_tokens=max_tokens)
        text = (resp.text or "").strip()
        if text:
            # Strip markdown fences if the model wraps the prompt.
            text = re.sub(r"^```.*?\n", "", text)
            text = re.sub(r"\n```$", "", text).strip()
            return text
    except Exception:
        pass
    return fallback_fn(subject)


def make_image_prompt(seed: Seed, ctx: "ActivityContext") -> str:
    subject = build_subject_context(seed, ctx)
    return _llm_prompt(ctx, subject, _IMAGE_SYSTEM, "image", _fallback_image)


def make_music_prompt(seed: Seed, ctx: "ActivityContext") -> str:
    subject = build_subject_context(seed, ctx)
    return _llm_prompt(ctx, subject, _MUSIC_SYSTEM, "music", _fallback_music)


def make_video_prompt(seed: Seed, ctx: "ActivityContext") -> str:
    subject = build_subject_context(seed, ctx)
    return _llm_prompt(ctx, subject, _VIDEO_SYSTEM, "video", _fallback_video)


def make_legacy_media_prompt(
    *,
    modality: str,
    seed_text: str,
    brief_body: str,
    cluster_label: str = "",
    llm=None,
) -> str:
    """Prompt for the legacy --generate path on research briefs (no full ActivityContext)."""

    @dataclass
    class _LegacyCtx:
        cluster_label: str
        parent_brief_md: str
        ground: bool = False
        llm: object | None = None

    seed = Seed(text=seed_text, source="legacy")
    ctx = _LegacyCtx(cluster_label=cluster_label, parent_brief_md=brief_body, llm=llm)
    subject = build_subject_context(seed, ctx)  # type: ignore[arg-type]

    if modality == "music":
        fb = _fallback_music
        systems = _MUSIC_SYSTEM
        label = "music"
    elif modality == "video":
        fb = _fallback_video
        systems = _VIDEO_SYSTEM
        label = "video"
    else:
        fb = _fallback_image
        systems = _IMAGE_SYSTEM
        label = "image"

    if llm is not None:
        return _llm_prompt(ctx, subject, systems, label, fb)  # type: ignore[arg-type]
    return fb(subject)
