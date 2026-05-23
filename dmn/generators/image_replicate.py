"""Replicate-hosted image generator (default: black-forest-labs/flux-schnell).

Uses the `replicate` python SDK. Reads `REPLICATE_API_TOKEN` from env. Model selectable via
`DMN_IMAGE_MODEL_REPLICATE`. Useful as a fallback when Nano Banana is unavailable.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")
DEFAULT_MODEL = "black-forest-labs/flux-schnell"


class ReplicateImageGenerator:
    """Image generation via the Replicate API."""

    modality = "image"
    name = "replicate_image"
    requires = ["REPLICATE_API_TOKEN", "replicate"]

    def available(self) -> bool:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            return False
        try:
            import replicate  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Run the configured Replicate model and save the first output to data/artifacts/."""
        import replicate  # type: ignore

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_id = os.environ.get("DMN_IMAGE_MODEL_REPLICATE", DEFAULT_MODEL)
        output = replicate.run(model_id, input={"prompt": prompt})
        png_bytes = _read_first_output(output)
        if not png_bytes:
            raise RuntimeError(f"Replicate model {model_id} returned no usable bytes")
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
            meta={"model": model_id},
        )


def _read_first_output(output) -> bytes:
    """Replicate returns a list of URLs or file-like objects; fetch the first as bytes."""
    if output is None:
        return b""
    items = list(output) if hasattr(output, "__iter__") and not isinstance(output, (bytes, str)) else [output]
    if not items:
        return b""
    first = items[0]
    if isinstance(first, (bytes, bytearray)):
        return bytes(first)
    if hasattr(first, "read"):
        try:
            return first.read()
        except Exception:
            pass
    if isinstance(first, str):
        try:
            import requests  # type: ignore

            r = requests.get(first, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception:
            return b""
    return b""


register(ReplicateImageGenerator())
