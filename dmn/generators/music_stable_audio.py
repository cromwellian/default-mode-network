"""Stability AI Stable Audio 2.0 music generator (v0.3).

Reads `STABILITY_API_KEY`. Calls the v2beta text-to-audio endpoint, which returns
audio bytes (mp3 by default). Saves to `data/artifacts/music_<ts>_<hash>.mp3`.

Docs: https://platform.stability.ai/docs/api-reference#tag/Audio
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")
ENDPOINT = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"
DEFAULT_SECONDS = 30
DEFAULT_OUTPUT_FORMAT = "mp3"


def _api_key() -> str:
    """Stability API key from env."""
    return os.environ.get("STABILITY_API_KEY") or ""


class StableAudioMusicGenerator:
    """Music generation via Stability AI's Stable Audio 2.0."""

    modality = "music"
    name = "stable_audio"
    requires = ["STABILITY_API_KEY", "requests"]

    def available(self) -> bool:
        if not _api_key():
            return False
        try:
            import requests  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """POST the prompt; save the returned audio bytes to data/artifacts/."""
        import requests  # type: ignore

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        seconds = int(kwargs.get("seconds") or DEFAULT_SECONDS)
        output_format = (kwargs.get("output_format") or DEFAULT_OUTPUT_FORMAT).lower()
        headers = {
            "Authorization": f"Bearer {_api_key()}",
            "Accept": "audio/*",
        }
        # Stability uses multipart/form-data; we don't need a file part, but `data=` works.
        files = {
            "prompt": (None, prompt),
            "duration": (None, str(seconds)),
            "output_format": (None, output_format),
        }
        t0 = time.time()
        resp = requests.post(ENDPOINT, headers=headers, files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Stable Audio HTTP {resp.status_code}: {resp.text[:300]}"
            )
        audio_bytes = resp.content
        if not audio_bytes:
            raise RuntimeError("Stable Audio returned no bytes")
        elapsed = round(time.time() - t0, 2)
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        ext = "mp3" if output_format == "mp3" else "wav"
        path = ARTIFACT_DIR / f"music_{int(time.time())}_{h}.{ext}"
        path.write_bytes(audio_bytes)
        return Artifact(
            modality="music",
            prompt=prompt,
            bytes_path=path,
            url=None,
            mime=f"audio/{ext}",
            generator=self.name,
            seconds=float(seconds),
            meta={"endpoint": ENDPOINT, "api_seconds": elapsed},
        )


register(StableAudioMusicGenerator())
