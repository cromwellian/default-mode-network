"""Google Lyria 2 music generation (the user said 'lyra' — clarifying: Lyria 2).

Lyria 2 is Google DeepMind's music generation model, exposed via Google AI / Vertex.
Public access is allow-listed as of early 2026, so this generator marks itself unavailable
by default and prints a friendly pointer if probed. When the public endpoint goes general,
this is where the call lands. See https://deepmind.google/technologies/lyria/.
"""
from __future__ import annotations

import os

from . import Artifact, register


def _api_key() -> str:
    """Return the Google API key from either env var name."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


class LyriaMusicGenerator:
    """Music generation via Google's Lyria 2 (currently allow-listed)."""

    modality = "music"
    name = "lyria"
    requires = ["GOOGLE_API_KEY (or GEMINI_API_KEY)", "google-genai", "Lyria allow-list access"]

    def available(self) -> bool:
        # Until the SDK exposes a stable music generation surface, treat as unavailable.
        # Flip this to a real probe once `client.models.generate_content` accepts a music
        # model name in your environment, or once Vertex AI exposes a published Lyria 2 model
        # ID via the public SDK.
        return False

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Raise a friendly error pointing at the Lyria allow-list page."""
        raise RuntimeError(
            "Lyria 2 requires allow-listed Google AI access; see "
            "https://deepmind.google/technologies/lyria/ . When the public SDK exposes "
            "music generation, drop the call into this stub."
        )


register(LyriaMusicGenerator())
