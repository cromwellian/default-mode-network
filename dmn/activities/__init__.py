"""Pluggable wander activities (v0.3): research, code_sketch, app_idea, ...

Each activity is a small module that exposes a single instance with a uniform interface.
The wander loop picks one per iteration (weighted by `--activity-mix`), runs it, and
treats the returned `ActivityResult` as the canonical brief artifact.

Sibling to `dmn.tools` (search backends) and `dmn.generators` (media producers). Tools
*find*, generators *make*, activities *do*.

Example:
    from dmn.activities import REGISTRY, pick_activity, parse_mix
    mix = parse_mix("research:5,code_sketch:2,app_idea:1")  # weighted dict
    activity = pick_activity(mix, rng, ctx)
    result = activity.run(seed, ctx)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from dmn.generators import Artifact
from dmn.seeds import Seed


@dataclass
class ActivityResult:
    """The output of one activity run: brief markdown body, artifacts, and optional execution log."""

    title: str
    body_md: str
    artifacts: list[Artifact] = field(default_factory=list)
    embedding_text: str = ""  # what to embed for dopamine scoring
    metadata: dict = field(default_factory=dict)
    execution: Optional[dict] = None  # {stdout, stderr, exit_code, duration_s, mode, timed_out}


@dataclass
class ActivityContext:
    """Everything an activity might need from the wander loop, in one bag."""

    llm: Any
    embed_fn: Callable[[list[str]], np.ndarray]
    clusters: list[dict]
    centroids: list[np.ndarray]
    recent_embs: list[np.ndarray]
    rng: random.Random
    dry_run: bool = False
    execute: bool = False
    sandbox: str = "subprocess"
    verbose: bool = False
    timeout_s: float = 30.0
    artifact_root: Path = Path("data/artifacts")
    code_budget: str = "small"


@runtime_checkable
class Activity(Protocol):
    """Activity interface. Implementations live in this package and self-register."""

    name: str
    requires_llm: bool
    requires_keys: list[str]
    requires_extras: list[str]

    def available(self, ctx: ActivityContext) -> bool: ...
    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult: ...


_REGISTRY: dict[str, "Activity"] = {}
REGISTRY = _REGISTRY  # public alias


def register(activity: "Activity") -> "Activity":
    """Register an activity instance under its `name`. Returns the instance for chaining."""
    _REGISTRY[activity.name] = activity
    return activity


def get(name: str) -> Optional["Activity"]:
    """Look up an activity by name."""
    return _REGISTRY.get(name)


def all_activities() -> list["Activity"]:
    """Return every registered activity (ignoring availability)."""
    return list(_REGISTRY.values())


def parse_mix(spec: str) -> dict[str, int]:
    """Parse a CSV / mix spec into a weighted dict.

    Examples:
        'research'                     -> {'research': 1}
        'research,code_sketch'         -> {'research': 1, 'code_sketch': 1}
        'research:5,code_sketch:2'     -> {'research': 5, 'code_sketch': 2}
    """
    if not spec:
        return {"research": 1}
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight = part.split(":", 1)
            try:
                w = max(0, int(weight.strip()))
            except ValueError:
                w = 1
            out[name.strip()] = w
        else:
            out[part] = 1
    return out or {"research": 1}


# Cluster keyword → activity boost. Lightweight nudge; the user's --activity-mix is
# always primary. Unmatched clusters get the unweighted mix.
_CLUSTER_BOOSTS: dict[str, list[str]] = {
    "image_riff": ["visual", "art", "design", "photo", "painting", "illustration", "icon"],
    "music_riff": ["music", "sound", "jazz", "rhythm", "polyrhythm", "lo-fi", "song"],
    "code_sketch": ["code", "programming", "lisp", "haskell", "python", "rust", "compiler"],
    "algorithm_explore": ["algorithm", "data structure", "optimization", "swarm", "graph"],
    "ml_experiment": ["ml", "neural", "transformer", "embedding", "gradient", "attention", "rlhf"],
    "app_idea": ["app", "product", "tool", "ux", "ui"],
}

_VISUAL_CUES = [
    "visual", "art", "artist", "painting", "drawing", "design", "photo", "image",
    "illustration", "media", "cinema", "film", "diagram", "simulation", "geometry",
    "map", "network", "graph", "architecture", "spatial", "color", "ui", "interface",
]

_AUDIO_CUES = [
    "music", "musical", "audio", "sound", "song", "rhythm", "tempo", "speech",
    "voice", "sonic", "instrument", "jazz", "album", "listening", "podcast",
]


def _boosted_weights(
    base: dict[str, int],
    cluster_label: str,
    *,
    boost: int = 2,
) -> dict[str, float]:
    """Apply a small additive boost to activities whose keywords match the cluster label."""
    label = (cluster_label or "").lower()
    out: dict[str, float] = {k: float(v) for k, v in base.items()}
    for activity_name, kws in _CLUSTER_BOOSTS.items():
        if activity_name not in out:
            continue
        if any(kw in label for kw in kws):
            out[activity_name] = out.get(activity_name, 0) + boost
    return out


def modality_weight_factor(
    activity_name: str,
    *,
    seed_text: str = "",
    cluster_label: str = "",
    forced_only: bool = False,
) -> float:
    """Return a heuristic multiplier for media activities.

    Image/music riffs are expensive and often silly on abstract topics. Keep them in the
    pool when explicitly requested, but strongly downweight them unless the seed or
    cluster text suggests that visual/audio form will explain the idea well.
    """
    if activity_name not in {"image_riff", "music_riff"}:
        return 1.0
    if forced_only:
        return 1.0
    text = f"{seed_text} {cluster_label}".lower()
    cues = _VISUAL_CUES if activity_name == "image_riff" else _AUDIO_CUES
    return 1.0 if any(cue in text for cue in cues) else 0.08


def pick_activity(
    mix: dict[str, int],
    ctx: ActivityContext,
    rng: random.Random,
    cluster_label: str = "",
    seed_text: str = "",
) -> Optional["Activity"]:
    """Sample an activity, weighted by `mix`, restricted to available ones.

    Falls back to the highest-weighted *available* activity if the random pick chose an
    unavailable one (e.g. the user listed `image_riff` but has no image API key set).
    """
    weights = _boosted_weights(mix, cluster_label) if cluster_label else {k: float(v) for k, v in mix.items()}
    positive = [name for name, w in mix.items() if w > 0]
    forced_only = len(positive) == 1 and positive[0] in {"image_riff", "music_riff"}
    candidates: list[tuple[str, float]] = []
    for name, w in weights.items():
        weight = float(w) * modality_weight_factor(
            name,
            seed_text=seed_text,
            cluster_label=cluster_label,
            forced_only=forced_only,
        )
        if weight <= 0:
            continue
        a = _REGISTRY.get(name)
        if not a:
            continue
        try:
            if not a.available(ctx):
                continue
        except Exception:
            continue
        candidates.append((name, weight))
    if not candidates:
        # Last-resort fallback: research must always be available (no keys, no extras).
        return _REGISTRY.get("research")
    total = sum(w for _, w in candidates)
    pick = rng.random() * total
    acc = 0.0
    for name, w in candidates:
        acc += w
        if pick < acc:
            return _REGISTRY[name]
    return _REGISTRY[candidates[-1][0]]


def available_activity_names(ctx: ActivityContext) -> list[str]:
    """Names of activities whose `available(ctx)` returns True (sorted alphabetically)."""
    out: list[str] = []
    for a in _REGISTRY.values():
        try:
            if a.available(ctx):
                out.append(a.name)
        except Exception:
            continue
    return sorted(out)


def _register_all() -> None:
    """Auto-import every activity submodule so each self-registers on import."""
    for mod in [
        "research",
        "code_sketch",
        "app_idea",
        "algorithm_explore",
        "ml_experiment",
        "image_riff",
        "music_riff",
        "video_riff",
        "web_app_sketch",
        "mood_journal",
    ]:
        try:
            import_module(f"dmn.activities.{mod}")
        except Exception:
            continue


_register_all()
