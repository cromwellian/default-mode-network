"""Sandboxed execution for code-generating activities (v0.3).

Three modes:
  - `auto`:       use Docker when available, otherwise subprocess.
  - `subprocess`: `subprocess.run(...)` with a strict env strip (no API keys, secrets,
    cloud creds visible to the child). Cheap convenience mode, no Docker dep.
  - `docker`:     `docker run --rm --network=none --read-only -v <tmp>:/work` against
    a stock `python:3.11-slim`. Strong isolation; activated when the user passes
    `--sandbox docker` and `docker` is on PATH.
  - `none`:       explicit "do not execute". Returns a sentinel result so callers can
    short-circuit without branching everywhere.

The env strip is the security-relevant part. We drop anything matching common
credential / secret patterns; see `_strip_env`. There's a small allow-list for things
the child genuinely needs (PATH, HOME, LANG, etc.) so subprocess Python can boot.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

# Anything matching these patterns is removed from the child's env. The strip is
# pessimistic on purpose — a generated script should never need an API key. If you
# legitimately need a key in the child, set it via `extra_env=`.
_STRIP_PATTERNS = [
    re.compile(r".*_API_KEY$", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_SECRET$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r".*_PASSPHRASE$", re.IGNORECASE),
    re.compile(r"^ANTHROPIC_.*", re.IGNORECASE),
    re.compile(r"^OPENAI_.*", re.IGNORECASE),
    re.compile(r"^GOOGLE_.*", re.IGNORECASE),
    re.compile(r"^GEMINI_.*", re.IGNORECASE),
    re.compile(r"^AZURE_.*", re.IGNORECASE),
    re.compile(r"^AWS_.*", re.IGNORECASE),
    re.compile(r"^GCP_.*", re.IGNORECASE),
    re.compile(r"^GCLOUD_.*", re.IGNORECASE),
    re.compile(r"^REPLICATE_.*", re.IGNORECASE),
    re.compile(r"^STABILITY_.*", re.IGNORECASE),
    re.compile(r"^SUNO_.*", re.IGNORECASE),
    re.compile(r"^HUGGINGFACE_.*", re.IGNORECASE),
    re.compile(r"^HF_.*", re.IGNORECASE),
    re.compile(r"^DMN_LLM_.*", re.IGNORECASE),
]

# These are kept so the subprocess can actually boot Python, find packages, and write
# UTF-8 output. Everything else is dropped if it matches a strip pattern OR retained
# only by being on this list.
_KEEP_KEYS = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "PYTHONPATH",
    "PYTHONIOENCODING",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "DMN_EMBEDDINGS",
}


def _strip_env(extra_env: Optional[dict] = None) -> dict:
    """Build a minimal env: keep only safe vars; never inherit credentials."""
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if any(p.match(k) for p in _STRIP_PATTERNS):
            continue
        if k in _KEEP_KEYS or k.startswith("PYTHON"):
            env[k] = v
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


def run_python(
    code_path: Path,
    *,
    mode: str = "subprocess",
    timeout: float = 30.0,
    extra_env: Optional[dict] = None,
) -> dict:
    """Execute `code_path` with the chosen sandbox; return a structured result.

    Returns:
        {
            "stdout": str, "stderr": str, "exit_code": int,
            "duration_s": float, "mode": str, "timed_out": bool,
        }
    """
    code_path = Path(code_path).resolve()
    mode = normalize_mode(mode)
    if mode == "none":
        return {
            "stdout": "",
            "stderr": "execution disabled (sandbox=none)",
            "exit_code": -1,
            "duration_s": 0.0,
            "mode": "none",
            "timed_out": False,
        }
    if not code_path.exists():
        return {
            "stdout": "",
            "stderr": f"code path not found: {code_path}",
            "exit_code": -2,
            "duration_s": 0.0,
            "mode": mode,
            "timed_out": False,
        }

    if mode == "docker":
        return _run_docker(code_path, timeout=timeout, extra_env=extra_env)
    return _run_subprocess(code_path, timeout=timeout, extra_env=extra_env)


def _run_subprocess(
    code_path: Path, *, timeout: float, extra_env: Optional[dict]
) -> dict:
    """Local-Python execution with a stripped env. The default mode."""
    import sys

    env = _strip_env(extra_env)
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(code_path.parent),
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = int(proc.returncode)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
        stderr += f"\n[sandbox] timed out after {timeout:.1f}s"
        exit_code = -3
    except Exception as exc:
        stdout = ""
        stderr = f"[sandbox] subprocess failed to start: {exc}"
        exit_code = -4
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration_s": round(time.time() - t0, 3),
        "mode": "subprocess",
        "timed_out": timed_out,
    }


def _run_docker(
    code_path: Path, *, timeout: float, extra_env: Optional[dict]
) -> dict:
    """Container execution with `--network=none` and a read-only root + writable /work mount."""
    if not shutil.which("docker"):
        return {
            "stdout": "",
            "stderr": "[sandbox] docker not found on PATH; pass --sandbox subprocess or install Docker",
            "exit_code": -5,
            "duration_s": 0.0,
            "mode": "docker",
            "timed_out": False,
        }
    work = code_path.parent.resolve()
    fname = code_path.name
    env = _strip_env(extra_env)
    env_args: list[str] = []
    for k, v in env.items():
        env_args.extend(["-e", f"{k}={v}"])

    cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "--read-only",
        "-v", f"{work}:/work:rw",
        "-w", "/work",
        *env_args,
        "python:3.11-slim",
        "python", f"/work/{fname}",
    ]
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = int(proc.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout = ""
        stderr = f"[sandbox] docker run timed out after {timeout:.1f}s"
        exit_code = -3
    except Exception as exc:
        stdout = ""
        stderr = f"[sandbox] docker invocation failed: {exc}"
        exit_code = -4
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration_s": round(time.time() - t0, 3),
        "mode": "docker",
        "timed_out": timed_out,
    }


def docker_available() -> bool:
    """True iff `docker` is on PATH and the daemon is reachable."""
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, text=True
        )
        return proc.returncode == 0
    except Exception:
        return False


def normalize_mode(mode: str) -> str:
    """Resolve sandbox mode aliases. `auto` prefers Docker when the daemon is reachable."""
    m = (mode or "auto").strip().lower()
    if m in {"none", "off", "disabled"}:
        return "none"
    if m == "auto":
        return "docker" if docker_available() else "subprocess"
    if m in {"docker", "subprocess"}:
        return m
    return "subprocess"
