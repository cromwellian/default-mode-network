"""Dry-run stub generators. Always available, write tiny placeholder files to disk.

Used when `explore.py --dry-run --generate` is invoked: lets the artifact pipeline
exercise the full path (file write, frontmatter wiring, markdown embed) with zero deps
and zero API keys.

- Image: a 1x1 PNG built from raw zlib + struct (no Pillow needed).
- Music: a 1-second 440Hz sine WAV via the stdlib `wave` module.
- Video: a tiny .txt placeholder (no stdlib mp4 writer; fine for a smoke test).
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
import time
import wave
import zlib
from pathlib import Path

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")


def _ensure_dir() -> None:
    """Create the artifacts dir if missing."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _stem(prompt: str, prefix: str) -> str:
    """Produce a unique-ish filename stem from a prompt."""
    h = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{int(time.time())}_{h}"


def _png_solid(rgba: tuple[int, int, int, int] = (40, 40, 40, 255)) -> bytes:
    """Return the raw bytes of a 1x1 PNG with the given RGBA pixel."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    raw = b"\x00" + bytes(rgba)  # filter byte + 4 channel bytes
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _wav_sine(seconds: float = 1.0, freq: int = 440, sr: int = 8000) -> bytes:
    """Return raw WAV bytes for a short sine tone."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        frames = bytearray()
        for i in range(n):
            v = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class DryRunImageGenerator:
    """Always-available image generator that writes a 1x1 PNG to data/artifacts/."""

    modality = "image"
    name = "dryrun_image"
    requires: list[str] = []

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Write a 1x1 PNG and return its Artifact."""
        _ensure_dir()
        path = ARTIFACT_DIR / f"{_stem(prompt, 'img')}.png"
        path.write_bytes(_png_solid())
        return Artifact(
            modality="image",
            prompt=prompt,
            bytes_path=path,
            url=None,
            mime="image/png",
            generator=self.name,
            seconds=None,
            meta={"note": "dry-run 1x1 placeholder"},
        )


class DryRunMusicGenerator:
    """Always-available music generator that writes a 1-second sine WAV."""

    modality = "music"
    name = "dryrun_music"
    requires: list[str] = []

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Write a 1-second sine WAV and return its Artifact."""
        _ensure_dir()
        path = ARTIFACT_DIR / f"{_stem(prompt, 'music')}.wav"
        path.write_bytes(_wav_sine())
        return Artifact(
            modality="music",
            prompt=prompt,
            bytes_path=path,
            url=None,
            mime="audio/wav",
            generator=self.name,
            seconds=1.0,
            meta={"note": "dry-run 440Hz sine placeholder"},
        )


class DryRunVideoGenerator:
    """Always-available video generator that writes a tiny .txt placeholder."""

    modality = "video"
    name = "dryrun_video"
    requires: list[str] = []

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Artifact:
        """Write a tiny .txt placeholder file and return its Artifact."""
        _ensure_dir()
        path = ARTIFACT_DIR / f"{_stem(prompt, 'video')}.txt"
        path.write_text(f"video placeholder for prompt: {prompt}\n")
        return Artifact(
            modality="video",
            prompt=prompt,
            bytes_path=path,
            url=None,
            mime="text/plain",
            generator=self.name,
            seconds=0.0,
            meta={"note": "dry-run text placeholder; real video gen lives in v0.2+"},
        )


register(DryRunImageGenerator())
register(DryRunMusicGenerator())
register(DryRunVideoGenerator())
