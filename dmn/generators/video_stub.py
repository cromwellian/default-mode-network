"""Placeholder for video generators (Veo 3 / Sora / Runway / Kling / Pika / ...).

Real video generation is intentionally deferred from v0.1.x: latency is awkward inside a
wander loop (a single Veo or Sora call can dominate the 12-min budget), and most providers
require either an allow-list or a webhook callback for async polling. This stub keeps the
modality wired into the registry so future adapters can drop in next to it.

To add a real backend: copy this file, set `available()` to check for the relevant API key,
implement `generate()`, and call `register(YourBackendVideoGenerator())` at module bottom.
"""
from __future__ import annotations

from . import Artifact, register


class StubVideoGenerator:
    """No-op video generator. Always unavailable; documents where Veo/Sora/Runway adapters land."""

    modality = "video"
    name = "video_stub"
    requires: list[str] = []

    def available(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Raise a friendly note that no real video backend is wired in yet."""
        raise RuntimeError(
            "No real video generator is wired in for v0.1.1. Drop a Veo / Sora / "
            "Runway / Kling / Pika adapter next to this file and call `register(...)`."
        )


register(StubVideoGenerator())
