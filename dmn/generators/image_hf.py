"""Hugging Face Inference API image generator (default: FLUX.1-schnell).

Reads `HF_TOKEN` or `HUGGINGFACE_API_KEY`. Uses `huggingface_hub.InferenceClient.text_to_image`.
Model override via `DMN_IMAGE_MODEL_HF`.
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


def _api_key() -> str:
    """Hugging Face token from either common env var name."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or ""


class HuggingFaceImageGenerator:
    """Image generation via the Hugging Face Inference API."""

    modality = "image"
    name = "hf_flux"
    requires = ["HF_TOKEN (or HUGGINGFACE_API_KEY)", "huggingface-hub"]

    def available(self) -> bool:
        if not _api_key():
            return False
        try:
            from huggingface_hub import InferenceClient  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Generate an image and save PNG bytes to data/artifacts/."""
        from huggingface_hub import InferenceClient  # type: ignore

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_id = os.environ.get("DMN_IMAGE_MODEL_HF", DEFAULT_MODEL)
        client = InferenceClient(token=_api_key())
        image = client.text_to_image(prompt, model=model_id)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        if not png_bytes:
            raise RuntimeError(f"HF model {model_id} returned no image bytes")
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


register(HuggingFaceImageGenerator())
