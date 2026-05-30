"""Runway text-to-video generator.

Uses Runway's REST API directly so we can use freshly released models before the
Python SDK catches up. Reads `RUN_API_KEY`, `RUNWAYML_API_SECRET`, or
`RUNWAY_API_KEY`; saves the completed MP4 to `data/artifacts/`.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import requests

from . import Artifact, register

ARTIFACT_DIR = Path("data/artifacts")
API_BASE = "https://api.dev.runwayml.com/v1"
API_VERSION = "2024-11-06"
DEFAULT_MODEL = "seedance2"
DEFAULT_RATIO = "1280:720"
DEFAULT_DURATION = 10


class RunwayVideoGenerator:
    """Video generation via Runway `/v1/text_to_video` with polling."""

    modality = "video"
    name = "runway_video"
    requires = ["RUN_API_KEY (or RUNWAYML_API_SECRET/RUNWAY_API_KEY)", "requests"]

    def available(self) -> bool:
        return bool(_api_key())

    def generate(self, prompt: str, **kwargs: Any) -> Artifact:
        """Create a Runway video task, poll it, download the first output MP4."""
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model = str(kwargs.get("model") or os.environ.get("DMN_VIDEO_RUNWAY_MODEL") or DEFAULT_MODEL)
        ratio = str(kwargs.get("ratio") or os.environ.get("DMN_VIDEO_RUNWAY_RATIO") or DEFAULT_RATIO)
        duration = _duration(kwargs.get("duration") or os.environ.get("DMN_VIDEO_SECONDS"))
        timeout_s = int(os.environ.get("DMN_VIDEO_RUNWAY_TIMEOUT", "900"))
        poll_s = float(os.environ.get("DMN_VIDEO_RUNWAY_POLL_SECONDS", "5"))

        payload = {
            "model": model,
            "promptText": _truncate_prompt(prompt),
            "ratio": ratio,
            "duration": duration,
        }
        t0 = time.time()
        task = _request("POST", f"{API_BASE}/text_to_video", json=payload)
        task_id = str(task.get("id") or "")
        if not task_id:
            raise RuntimeError(f"Runway returned no task id: {task}")

        completed = _poll_task(task_id, timeout_s=timeout_s, poll_s=poll_s)
        output_url = _first_output_url(completed)
        if not output_url:
            raise RuntimeError(f"Runway task {task_id} succeeded without output URL: {completed}")

        video_bytes = _download(output_url)
        elapsed = round(time.time() - t0, 2)
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        path = ARTIFACT_DIR / f"video_{int(time.time())}_{h}.mp4"
        path.write_bytes(video_bytes)
        return Artifact(
            modality="video",
            prompt=prompt,
            bytes_path=path,
            url=output_url,
            mime="video/mp4",
            generator=self.name,
            seconds=float(duration),
            meta={
                "model": model,
                "ratio": ratio,
                "task_id": task_id,
                "api_seconds": elapsed,
                "provider_url": output_url,
            },
        )


def _api_key() -> str:
    return (
        os.environ.get("RUN_API_KEY")
        or os.environ.get("RUNWAYML_API_SECRET")
        or os.environ.get("RUNWAY_API_KEY")
        or ""
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "X-Runway-Version": API_VERSION,
    }


def _request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    resp = requests.request(method, url, headers=_headers(), timeout=60, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = {"body": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(f"Runway API {resp.status_code}: {data}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Runway returned unexpected response: {data!r}")
    return data


def _poll_task(task_id: str, *, timeout_s: int, poll_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = _request("GET", f"{API_BASE}/tasks/{task_id}")
        status = str(last.get("status") or "").upper()
        if status == "SUCCEEDED":
            return last
        if status in {"FAILED", "CANCELLED", "CANCELED"}:
            failure = last.get("failure") or last.get("error") or last
            raise RuntimeError(f"Runway task {task_id} {status.lower()}: {failure}")
        time.sleep(poll_s)
    raise TimeoutError(f"Runway task {task_id} did not finish within {timeout_s}s: {last}")


def _first_output_url(task: dict[str, Any]) -> str:
    output = task.get("output")
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("url") or first.get("uri") or "")
    if isinstance(output, str):
        return output
    return ""


def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.content


def _duration(value: Any) -> int:
    try:
        duration = int(value or DEFAULT_DURATION)
    except Exception:
        duration = DEFAULT_DURATION
    return max(5, min(15, duration))


def _truncate_prompt(prompt: str) -> str:
    text = " ".join((prompt or "").split())
    return text[:1000]


register(RunwayVideoGenerator())
