"""Modal REST API for DMN profile provisioning and remote wanders."""

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "default-mode-network"
PROFILES_MOUNT = "/profiles"
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)

profile_volume = modal.Volume.from_name("dmn-profiles", create_if_missing=True)

# Core DMN image — sentence-transformers + torch make this heavy but required for taste scoring.
dmn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "python-multipart>=0.0.9",
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "arxiv>=2.1.0",
        "wikipedia-api>=0.7.0",
        "duckduckgo-search>=6.0.0",
        "requests>=2.31.0",
        "sentence-transformers>=3.0.0",
        "scikit-learn>=1.4.0",
        "hdbscan>=0.8.33",
        "numpy>=1.26.0",
        "rich>=13.7.0",
        "typer>=0.12.0",
        "python-dotenv>=1.0.0",
        "torch>=2.2.0",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(REPO_ROOT / "dmn"), remote_path="/root/dmn")
    .add_local_file(str(REPO_ROOT / "wander.py"), remote_path="/root/wander.py")
    .add_local_file(str(REPO_ROOT / "explore.py"), remote_path="/root/explore.py")
    .add_local_file(str(REPO_ROOT / "prepare.py"), remote_path="/root/prepare.py")
    .add_local_file(str(REPO_ROOT / "profile.py"), remote_path="/root/profile.py")
    .add_local_file(str(REPO_ROOT / "journal.py"), remote_path="/root/journal.py")
    .add_local_dir(str(REPO_ROOT / "modal_api"), remote_path="/root/modal_api")
)

# vLLM image for local-model inference (Qwen, Gemma, etc.)
VLLM_MODEL_PRESETS: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
}

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub>=0.26.0")
)


def _validate_profile_db(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        interest_count = conn.execute("SELECT COUNT(*) FROM interests").fetchone()[0]
        cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        if interest_count == 0:
            raise ValueError("SQLite profile has no interests — run prepare.py locally first")
        return {
            "interest_count": int(interest_count),
            "cluster_count": int(cluster_count),
        }
    finally:
        conn.close()


def _resolve_llm(config_data: dict[str, Any]) -> dict[str, str | None]:
    """Map API llm block to DMN env overrides."""
    llm = config_data.get("llm") or {}
    provider = llm.get("provider")
    model = llm.get("model")
    preset = llm.get("preset")
    base_url = llm.get("base_url")

    if preset:
        model = VLLM_MODEL_PRESETS.get(preset, preset)
        provider = provider or "vllm"

    if provider == "vllm" and not base_url:
        # Wander function will inject the live vLLM web URL before running.
        pass

    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_base_url": base_url,
    }


@app.function(
    image=vllm_image,
    gpu="A10G",
    timeout=3600,
    scaledown_window=600,
    secrets=[modal.Secret.from_name("dmn-env")],
)
@modal.web_server(port=8000, startup_timeout=600)
def vllm_server() -> None:
    """OpenAI-compatible vLLM server. Model id via VLLM_MODEL env (set per invocation)."""
    import os
    import subprocess
    import sys

    model_id = os.environ.get("VLLM_MODEL", VLLM_MODEL_PRESETS["qwen2.5-7b"])
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            "auto",
        ]
    )


@app.function(
    image=dmn_image,
    secrets=[modal.Secret.from_name("dmn-env")],
    volumes={PROFILES_MOUNT: profile_volume},
    timeout=3600,
    memory=8192,
    cpu=4,
)
def run_wander_job(
    user_id: str,
    profile_id: str,
    wander_config: dict[str, Any],
    base_url: str,
    vllm_model: str | None = None,
) -> dict[str, Any]:
    """Execute a wander in an isolated workspace and persist outputs to the profile volume."""
    import sys

    sys.path.insert(0, "/root")
    from dmn.api_paths import profile_dir as tenant_profile_dir
    from dmn.api_response import build_wander_response
    from dmn.wander_config import WanderConfig
    from dmn.wander_runner import _collect_files, run_wander

    profile_volume.reload()

    profile_path = tenant_profile_dir(PROFILES_MOUNT, user_id, profile_id)
    db_path = profile_path / "dmn.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"profile not found: user={user_id} profile={profile_id}")

    llm_overrides = _resolve_llm(wander_config)
    wander_fields = {
        k: v
        for k, v in wander_config.items()
        if k not in ("llm",)
    }
    wander_fields.update({k: v for k, v in llm_overrides.items() if v is not None})
    config = WanderConfig.from_dict(wander_fields)

    if config.llm_provider == "vllm" and not config.llm_base_url:
        model_id = vllm_model or config.llm_model or VLLM_MODEL_PRESETS["qwen2.5-7b"]
        vllm_fn = vllm_server.with_options(env={"VLLM_MODEL": model_id})
        vllm_url = vllm_fn.get_web_url()
        config.llm_base_url = vllm_url.rstrip("/") + "/v1"
        if not config.llm_model:
            config.llm_model = model_id

    run_work_dir = profile_path / "work" / uuid.uuid4().hex[:12]
    if run_work_dir.exists():
        shutil.rmtree(run_work_dir)

    timeout_s = None
    if config.minutes and config.minutes > 0:
        timeout_s = config.minutes * 60 + 120

    result = run_wander(
        config,
        profile_db=db_path,
        work_dir=run_work_dir,
        timeout_s=timeout_s,
    )

    run_dir = profile_path / "runs" / result.run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if not run_work_dir.exists():
        raise FileNotFoundError(f"wander work dir missing before finalize: {run_work_dir}")
    shutil.copytree(run_work_dir, run_dir)
    shutil.rmtree(run_work_dir)

    meta = {
        "user_id": user_id,
        "run_id": result.run_id,
        "profile_id": profile_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "brief_count": result.brief_count,
            "best_score": result.best_score,
            "duration_seconds": result.duration_seconds,
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    result.work_dir = str(run_dir)
    result.files = _collect_files(run_dir)  # refresh paths after rename
    profile_volume.commit()

    return build_wander_response(
        result, user_id=user_id, profile_id=profile_id, base_url=base_url
    )


@app.function(
    image=dmn_image,
    secrets=[modal.Secret.from_name("dmn-env")],
    volumes={PROFILES_MOUNT: profile_volume},
    timeout=900,
)
@modal.asgi_app()
def api():
    """FastAPI REST surface for profiles, wanders, and artifact fetch."""
    import sys

    sys.path.insert(0, "/root")
    from modal_api.web import create_web_app

    return create_web_app(
        app_name=APP_NAME,
        profiles_mount=PROFILES_MOUNT,
        profile_volume=profile_volume,
        run_wander_job=run_wander_job,
        validate_profile_db=_validate_profile_db,
        vllm_presets=VLLM_MODEL_PRESETS,
    )
