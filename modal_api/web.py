"""FastAPI routes for the DMN Modal API (module-level routes for FastAPI type resolution)."""
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import starlette.requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dmn.api_paths import (
    artifact_url,
    profile_dir as tenant_profile_dir,
    run_dir as tenant_run_dir,
    user_dir as tenant_user_dir,
    validate_user_id,
)
from dmn.wander_config import WanderConfig

web = FastAPI(
    title="Default Mode Network API",
    description=(
        "Provision taste profiles and trigger remote wanders on Modal. "
        "All profiles and artifacts are scoped under a caller-provided user_id."
    ),
    version="0.2.4",
)

# Injected by create_web_app() before the ASGI app is served.
APP_NAME: str = "default-mode-network"
PROFILES_MOUNT: str = "/profiles"
PROFILE_VOLUME: Any = None
RUN_WANDER_JOB: Any = None
VALIDATE_PROFILE_DB: Callable[[Path], dict[str, Any]] | None = None
VLLM_PRESETS: dict[str, str] = {}


class LLMConfig(BaseModel):
    provider: str | None = Field(
        default=None,
        description="anthropic | openai | ollama | lmstudio | vllm | stub",
    )
    model: str | None = Field(default=None, description="Model id or HuggingFace repo")
    preset: str | None = Field(default=None, description="vLLM preset name")
    base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL (for vllm/ollama/lmstudio)",
    )


class WanderRequest(BaseModel):
    minutes: float = 12.0
    iterations: int | None = None
    max_depth: int = 4
    children_per_expansion: int = 3
    root_count: int = 3
    restart_prob: float = 0.08
    margin: float = 0.05
    min_absolute: float = 0.15
    similarity_threshold: float = 0.95
    seed_text: str | None = None
    activities: str = "research"
    activity_mix: str | None = None
    execute: bool = False
    no_execute: bool = False
    sandbox: str = "auto"
    dry_run: bool = False
    generate: bool = False
    modalities: str = "image,music,video"
    resume: bool = False
    beam_width: int = 0
    patience: int = 0
    until_dopamine: float | None = None
    min_improvement: float = 0.01
    explore: float = 0.05
    ground: bool = True
    report: bool = True
    code_budget: str = "small"
    verbose: bool = False
    llm: LLMConfig | None = None

    def to_runner_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        if self.llm:
            data["llm"] = self.llm.model_dump(exclude_none=True)
        return data


@web.get("/v1/health")
def health():
    return {"status": "ok", "app": APP_NAME, "api_version": "0.2.4"}


@web.get("/v1/models")
def list_models():
    return {
        "vllm_presets": VLLM_PRESETS,
        "llm_providers": ["anthropic", "openai", "ollama", "lmstudio", "vllm", "stub"],
    }


@web.post("/v1/users/{user_id}/profiles")
async def create_profile(user_id: str, request: starlette.requests.Request):
    try:
        user_id = validate_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        raise HTTPException(status_code=400, detail="missing form field 'file'")

    profile_id = uuid.uuid4().hex[:12]
    dest = tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    dest.mkdir(parents=True, exist_ok=True)
    db_path = dest / "dmn.sqlite"

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty upload")
    db_path.write_bytes(content)

    if VALIDATE_PROFILE_DB is None:
        raise HTTPException(status_code=500, detail="server not initialized")

    try:
        stats = VALIDATE_PROFILE_DB(db_path)
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e

    original_filename = getattr(upload, "filename", None)
    meta = {
        "user_id": user_id,
        "profile_id": profile_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_filename": original_filename,
        **stats,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    PROFILE_VOLUME.commit()

    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "user_id": user_id,
            "profile_id": profile_id,
            "stats": stats,
            "url": f"{base}/v1/users/{user_id}/profiles/{profile_id}",
        }
    )


@web.get("/v1/users/{user_id}/profiles")
def list_profiles(user_id: str):
    try:
        user_id = validate_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    user_root = tenant_user_dir(PROFILES_MOUNT, user_id)
    profiles = []
    if user_root.exists():
        for p in sorted(user_root.iterdir()):
            if not p.is_dir():
                continue
            meta_file = p / "meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            profiles.append({"profile_id": p.name, **meta})
    return {"user_id": user_id, "profiles": profiles}


@web.get("/v1/users/{user_id}/profiles/{profile_id}")
def get_profile(user_id: str, profile_id: str, request: starlette.requests.Request):
    try:
        dest = tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not (dest / "dmn.sqlite").is_file():
        raise HTTPException(status_code=404, detail="profile not found")
    meta_file = dest / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    runs_dir = dest / "runs"
    runs = sorted(r.name for r in runs_dir.iterdir()) if runs_dir.exists() else []
    base = str(request.base_url).rstrip("/")
    return {
        "user_id": user_id,
        "profile_id": profile_id,
        "meta": meta,
        "runs": runs,
        "url": f"{base}/v1/users/{user_id}/profiles/{profile_id}",
    }


@web.delete("/v1/users/{user_id}/profiles/{profile_id}")
def delete_profile(user_id: str, profile_id: str):
    try:
        dest = tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not dest.exists():
        raise HTTPException(status_code=404, detail="profile not found")
    shutil.rmtree(dest)
    PROFILE_VOLUME.commit()
    return {"status": "deleted", "user_id": user_id, "profile_id": profile_id}


@web.post("/v1/users/{user_id}/profiles/{profile_id}/wander")
def start_wander(
    user_id: str,
    profile_id: str,
    body: WanderRequest,
    request: starlette.requests.Request,
):
    try:
        tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    dest = tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    if not (dest / "dmn.sqlite").is_file():
        raise HTTPException(status_code=404, detail="profile not found")

    base_url = str(request.base_url).rstrip("/")
    config_dict = body.to_runner_dict()
    llm = config_dict.get("llm") or {}
    vllm_model = None
    if llm.get("preset"):
        vllm_model = VLLM_PRESETS.get(llm["preset"], llm["preset"])
    elif llm.get("model") and (llm.get("provider") == "vllm" or llm.get("preset")):
        vllm_model = llm.get("model")

    if RUN_WANDER_JOB is None:
        raise HTTPException(status_code=500, detail="server not initialized")

    result = RUN_WANDER_JOB.remote(
        user_id,
        profile_id,
        config_dict,
        base_url,
        vllm_model,
    )
    return JSONResponse(result)


@web.get("/v1/users/{user_id}/profiles/{profile_id}/runs/{run_id}")
def get_run(
    user_id: str,
    profile_id: str,
    run_id: str,
    request: starlette.requests.Request,
):
    try:
        run_path = tenant_run_dir(PROFILES_MOUNT, user_id, profile_id, run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not run_path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    meta_file = run_path / "run_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {"run_id": run_id}
    base = str(request.base_url).rstrip("/")

    files = []
    for p in sorted(run_path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(run_path))
            files.append(
                {
                    "path": rel,
                    "url": artifact_url(base, user_id, profile_id, run_id, rel),
                }
            )
    return {
        "user_id": user_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "meta": meta,
        "files": files,
    }


@web.get("/v1/users/{user_id}/profiles/{profile_id}/runs/{run_id}/files/{file_path:path}")
def fetch_file(user_id: str, profile_id: str, run_id: str, file_path: str):
    try:
        run_path = tenant_run_dir(PROFILES_MOUNT, user_id, profile_id, run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    target = (run_path / file_path).resolve()
    if not str(target).startswith(str(run_path.resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@web.get("/v1/wander/params")
def wander_params():
    fields = WanderConfig().__dataclass_fields__
    return {
        param: {
            "default": field.default,
            "type": str(field.type),
        }
        for param, field in fields.items()
    }


def create_web_app(
    *,
    app_name: str,
    profiles_mount: str,
    profile_volume: Any,
    run_wander_job: Any,
    validate_profile_db: Callable[[Path], dict[str, Any]],
    vllm_presets: dict[str, str],
) -> FastAPI:
    global APP_NAME, PROFILES_MOUNT, PROFILE_VOLUME, RUN_WANDER_JOB
    global VALIDATE_PROFILE_DB, VLLM_PRESETS

    APP_NAME = app_name
    PROFILES_MOUNT = profiles_mount
    PROFILE_VOLUME = profile_volume
    RUN_WANDER_JOB = run_wander_job
    VALIDATE_PROFILE_DB = validate_profile_db
    VLLM_PRESETS = vllm_presets
    return web
