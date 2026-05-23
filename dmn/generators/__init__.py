"""Pluggable generative-media tools (image / music / video / text).

Sibling to `dmn.tools` but with a different interface: tools *find* things on the public
internet; generators *make* artifacts. Each generator is an instance with `.modality`,
`.name`, `.requires`, `.available()`, and `.generate(prompt, **kwargs) -> Artifact`.

Generators are OFF by default — wire them on in `explore.py` via `--generate`. The artifact's
embedding is NOT yet folded into the dopamine score (that's a v0.2 item — would need CLIP for
images, CLAP for audio).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Artifact:
    """A normalized generative-media output shared across all generator backends."""

    modality: str  # "image" | "music" | "video" | "text"
    prompt: str
    bytes_path: Optional[Path]
    url: Optional[str]
    mime: str
    generator: str
    seconds: Optional[float] = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Generator interface. Implementations live in this package and self-register."""

    modality: str
    name: str
    requires: list[str]

    def available(self) -> bool: ...
    def generate(self, prompt: str, **kwargs: Any) -> Artifact: ...


_REGISTRY: dict[str, "Generator"] = {}
REGISTRY = _REGISTRY  # public alias for ergonomic imports


def register(gen: "Generator") -> "Generator":
    """Register a generator instance under its `name`. Returns the instance for chaining."""
    _REGISTRY[gen.name] = gen
    return gen


def get(name: str) -> Optional["Generator"]:
    """Look up a registered generator by name."""
    return _REGISTRY.get(name)


def all_generators() -> list["Generator"]:
    """Return every registered generator (ignoring availability)."""
    return list(_REGISTRY.values())


def available_for(modality: str) -> list["Generator"]:
    """Generators of a given modality whose `available()` currently returns True."""
    return [g for g in _REGISTRY.values() if g.modality == modality and g.available()]


# String-matching heuristic: nudge a cluster's label toward a likely modality.
_KEYWORDS = {
    "image": [
        "visual", "art", "design", "photo", "painting", "color", "icon", "architecture",
        "drawing", "illustration", "renaissance",
    ],
    "music": [
        "music", "sound", "album", "jazz", "song", "rhythm", "polyrhythm", "scrobble",
        "lo-fi", "field recording",
    ],
    "video": ["film", "movie", "video", "cinema", "documentary"],
}


def pick_for_cluster(
    cluster_label: str, dry_run: bool = False
) -> list["Generator"]:
    """Heuristic: pick generators whose modality matches keywords in a cluster's label.

    Returns *available* generators only. In `dry_run` mode, restricts to the dry-run stubs
    so the pipeline can exercise the artifact path without any API keys.
    """
    label = (cluster_label or "").lower()
    modalities: list[str] = []
    for modality, kws in _KEYWORDS.items():
        if any(kw in label for kw in kws):
            modalities.append(modality)
    if not modalities:
        modalities = ["image"]  # default modality if nothing matches

    out: list[Generator] = []
    for modality in modalities:
        gens = available_for(modality)
        if dry_run:
            gens = [g for g in gens if g.name.startswith("dryrun_")]
        else:
            gens = [g for g in gens if not g.name.startswith("dryrun_")]
        out.extend(gens)
    return out


def _register_all() -> None:
    """Auto-import every generator submodule so they self-register on import."""
    from importlib import import_module

    for mod in [
        "_dry_run",
        "image_nano_banana",
        "image_replicate",
        "music_lyria",
        "music_stable_audio",
        "music_suno",
        "video_stub",
        "video_replicate",
    ]:
        try:
            import_module(f"dmn.generators.{mod}")
        except Exception:
            continue


_register_all()
