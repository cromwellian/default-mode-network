"""Google Gemini 2.5 Flash Image (a.k.a. 'Nano Banana') image generator.

Uses the `google-genai` SDK. Reads `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) from env.
Saves the returned PNG to `data/artifacts/img_<ts>_<hash>.png`. Gracefully marks itself
unavailable if the SDK isn't installed or no key is set.
"""
from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")
MODEL = "gemini-2.5-flash-image"


def _api_key() -> str:
    """Return the Google API key from either env var name."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


class NanoBananaImageGenerator:
    """Image generation via Gemini 2.5 Flash Image."""

    modality = "image"
    name = "nano_banana"
    requires = ["GOOGLE_API_KEY (or GEMINI_API_KEY)", "google-genai"]

    def available(self) -> bool:
        if not _api_key():
            return False
        try:
            import google.genai  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Generate an image and save the first returned PNG to data/artifacts/."""
        from google import genai  # type: ignore

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        client = genai.Client(api_key=_api_key())
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        png_bytes = _extract_image_bytes(resp)
        if not png_bytes:
            raise RuntimeError("Nano Banana returned no image bytes")
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        path = ARTIFACT_DIR / f"img_{int(time.time())}_{h}.png"
        path.write_bytes(png_bytes)
        return Artifact(
            modality="image",
            prompt=prompt,
            bytes_path=path,
            url=None,
            mime="image/png",
            generator=self.name,
            seconds=None,
            meta={"model": MODEL},
        )


def _extract_image_bytes(resp) -> bytes:
    """Walk the google-genai response shape to find the first image part's raw bytes."""
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if data is None:
                continue
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data)
                except Exception:
                    continue
    return b""


register(NanoBananaImageGenerator())
