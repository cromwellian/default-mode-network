"""Multi-tenant path helpers for the DMN REST API."""
from __future__ import annotations

import re
from pathlib import Path

USER_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
PROFILE_ID_RE = re.compile(r"^[a-f0-9]{12}$")
RUN_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def validate_user_id(user_id: str) -> str:
    """Reject path traversal and unsafe tenant ids."""
    user_id = (user_id or "").strip()
    if not USER_ID_RE.fullmatch(user_id):
        raise ValueError(
            "user_id must be 1-64 chars, start with alphanumeric, "
            "and contain only letters, digits, '.', '_', or '-'"
        )
    return user_id


def validate_profile_id(profile_id: str) -> str:
    profile_id = (profile_id or "").strip()
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("profile_id must be a 12-char hex id")
    return profile_id


def validate_run_id(run_id: str) -> str:
    run_id = (run_id or "").strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a 12-char hex id")
    return run_id


def user_dir(mount: str | Path, user_id: str) -> Path:
    return Path(mount) / validate_user_id(user_id)


def profile_dir(mount: str | Path, user_id: str, profile_id: str) -> Path:
    return user_dir(mount, user_id) / validate_profile_id(profile_id)


def run_dir(mount: str | Path, user_id: str, profile_id: str, run_id: str) -> Path:
    return profile_dir(mount, user_id, profile_id) / "runs" / validate_run_id(run_id)


def artifact_url(
    base_url: str,
    user_id: str,
    profile_id: str,
    run_id: str,
    rel_path: str,
) -> str:
    base = base_url.rstrip("/")
    clean = rel_path.lstrip("/")
    uid = validate_user_id(user_id)
    pid = validate_profile_id(profile_id)
    rid = validate_run_id(run_id)
    return f"{base}/v1/users/{uid}/profiles/{pid}/runs/{rid}/files/{clean}"
