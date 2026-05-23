"""Suno music generation stub (v0.3).

Suno doesn't have a stable, fully-public REST API as of v0.3 — community wrappers exist
but their auth surface shifts. This module is therefore a polite "available iff
`SUNO_API_KEY` is set AND a Suno SDK is importable" stub. When the public API
stabilizes, drop the actual call into `generate()`.
"""
from __future__ import annotations

import os

from . import Artifact, register


def _api_key() -> str:
    """Suno API key from env (whatever the eventual official name turns out to be)."""
    return os.environ.get("SUNO_API_KEY") or ""


class SunoMusicGenerator:
    """Music generation via Suno (gated; not yet wired to a public endpoint)."""

    modality = "music"
    name = "suno"
    requires = ["SUNO_API_KEY", "(unstable upstream API)"]

    def available(self) -> bool:
        # Even with a key, treat as unavailable until the upstream API is stable.
        if not _api_key():
            return False
        return False

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Raise a friendly hint pointing at the upstream status."""
        raise RuntimeError(
            "Suno music generation requires a stable upstream API; this stub is "
            "available for env-gating only. Drop the real call in once the upstream "
            "API is public."
        )


register(SunoMusicGenerator())
