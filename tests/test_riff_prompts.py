"""Tests for subject-aware riff prompt builders."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dmn.activities.riff_prompts import (
    build_subject_context,
    classify_subject,
    load_parent_brief_md,
    make_image_prompt,
    make_legacy_media_prompt,
    make_music_prompt,
)
from dmn.seeds import Seed


def test_classify_subject_technical():
    assert classify_subject(
        "How do transformer attention heads route information?",
        brief_md="Neural network architecture with embedding layers and gradient flow.",
        cluster_label="ml systems",
    ) == "technical"


def test_classify_subject_artistic():
    assert classify_subject(
        "What made Coltrane's Giant Steps changes so radical?",
        brief_md="Jazz harmony and improvisation in late 1950s bebop.",
        cluster_label="music history",
    ) == "artistic"


def test_build_subject_context_uses_parent_brief():
    @dataclass
    class Ctx:
        cluster_label: str = "neural networks"
        parent_brief_md: str = (
            "## Overview\n\n"
            "Attention maps show which tokens attend to which others in a transformer block."
        )
        ground: bool = False

    seed = Seed(text="Explain multi-head attention", source="test")
    subject = build_subject_context(seed, Ctx())
    assert subject.mode == "technical"
    assert "Attention maps" in subject.brief_excerpt
    assert subject.cluster_label == "neural networks"


def test_fallback_image_technical_mentions_diagram():
    @dataclass
    class Ctx:
        cluster_label: str = ""
        parent_brief_md: str = ""
        ground: bool = False
        llm: object = None

    seed = Seed(text="gradient descent optimization algorithm", source="test")
    prompt = make_image_prompt(seed, Ctx())
    lower = prompt.lower()
    assert "diagram" in lower or "infographic" in lower or "labeled" in lower
    assert "surreal" not in lower or "not surreal" in lower


def test_make_image_prompt_uses_llm_when_available():
    class StubLLM:
        def complete(self, *, system, user, max_tokens):
            assert "EXPLANATORY" in system or "illustrate" in system.lower()
            assert "gradient descent" in user
            return type("R", (), {"text": "Labeled diagram of gradient descent on a loss surface."})()

    @dataclass
    class Ctx:
        cluster_label: str = "optimization"
        parent_brief_md: str = "Gradient descent iteratively steps along the negative gradient."
        ground: bool = False
        llm: object = field(default_factory=StubLLM)

    seed = Seed(text="gradient descent", source="test")
    assert "Labeled diagram" in make_image_prompt(seed, Ctx())


def test_make_legacy_media_prompt_without_llm():
    prompt = make_legacy_media_prompt(
        modality="music",
        seed_text="ocean currents",
        brief_body="Thermohaline circulation moves deep water across basins.",
        cluster_label="earth science",
        llm=None,
    )
    assert "Thermohaline" in prompt or "ocean" in prompt.lower()
    assert "Instrumental" in prompt or "instrument" in prompt.lower()


def test_load_parent_brief_md(tmp_path: Path):
    body = "---\ntitle: x\n---\n\nParent brief about quantum tunneling."
    p = tmp_path / "brief.md"
    p.write_text(body, encoding="utf-8")
    md = load_parent_brief_md({"path": str(p)})
    assert "quantum tunneling" in md


def test_make_music_prompt_on_topic():
    class StubLLM:
        def complete(self, *, system, user, max_tokens):
            return type("R", (), {"text": "Slow jazz piano with brushed drums, 72 bpm, melancholy."})()

    @dataclass
    class Ctx:
        cluster_label: str = "jazz history"
        parent_brief_md: str = "Coltrane's harmonic substitutions on Giant Steps."
        ground: bool = False
        llm: object = field(default_factory=StubLLM)

    seed = Seed(text="Giant Steps changes", source="test")
    out = make_music_prompt(seed, Ctx())
    assert "jazz" in out.lower() or "Coltrane" in out or "72" in out
