"""Wander session configuration — mirrors `wander.py` CLI flags for programmatic use."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class WanderConfig:
    """All knobs exposed by `wander.py`, usable from CLI or REST API."""

    minutes: float = 12.0
    iterations: Optional[int] = None
    max_depth: int = 4
    children_per_expansion: int = 3
    root_count: int = 3
    restart_prob: float = 0.08
    margin: float = 0.05
    min_absolute: float = 0.15
    similarity_threshold: float = 0.95
    seed_text: Optional[str] = None
    activities: str = "research"
    activity_mix: Optional[str] = None
    execute: bool = False
    no_execute: bool = False
    sandbox: str = "auto"
    dry_run: bool = False
    generate: bool = False
    modalities: str = "image,music,video"
    resume: bool = False
    beam_width: int = 0
    patience: int = 0
    until_dopamine: Optional[float] = None
    min_improvement: float = 0.01
    explore: float = 0.05
    ground: bool = True
    report: bool = True
    code_budget: str = "small"
    verbose: bool = False
    # LLM overrides (applied to os.environ for the session)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WanderConfig":
        """Build from a JSON body; ignores unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WanderResult:
    """Structured output from a completed wander session."""

    run_id: str
    work_dir: str
    brief_count: int
    best_score: float
    pruned_count: int
    leaf_count: int
    open_count: int
    duration_seconds: float
    patience_triggered: bool
    beam_pruned: int
    briefs: list[dict[str, Any]] = field(default_factory=list)
    report_path: Optional[str] = None
    tree_path: Optional[str] = None
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
