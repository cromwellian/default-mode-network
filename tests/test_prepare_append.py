"""prepare appends + re-clusters by default; --replace starts fresh (issue #33)."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def run_prepare(cwd: Path, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "DMN_LLM_PROVIDER": "stub",
        "DMN_EMBEDDINGS": "hash",
    }
    return subprocess.run(
        [sys.executable, str(REPO / "prepare.py"), *args],
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )


def counts(cwd: Path) -> tuple[int, int]:
    conn = sqlite3.connect(cwd / "data" / "dmn.sqlite")
    n_int = conn.execute("SELECT COUNT(*) FROM interests").fetchone()[0]
    n_clu = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    conn.close()
    return n_int, n_clu


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    r = run_prepare(tmp_path, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    n, _ = counts(tmp_path)
    assert n == 12  # synthetic seed profile
    return tmp_path


def test_append_is_the_default(workspace: Path):
    r = run_prepare(
        workspace, "--interactive", stdin="coffee roasting\nkintsugi\n\n\n\n\n"
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Appending" in r.stdout
    n, n_clu = counts(workspace)
    assert n == 14  # 12 synthetic + 2 new — nothing wiped
    assert n_clu >= 2  # clusters rebuilt over the union


def test_append_skips_known_duplicates(workspace: Path):
    run_prepare(workspace, "--interactive", stdin="coffee roasting\n\n\n\n\n\n")
    r = run_prepare(workspace, "--interactive", stdin="coffee roasting\n\n\n\n\n\n")
    assert r.returncode == 0
    assert "Nothing new to add" in r.stdout
    assert counts(workspace)[0] == 13


def test_replace_starts_fresh(workspace: Path):
    r = run_prepare(
        workspace, "--interactive", "--replace", stdin="just one thing\n\n\n\n\n\n"
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "starting fresh" in r.stdout
    assert counts(workspace)[0] == 1


def test_interrupted_rebuild_blocks_wandering(workspace: Path):
    import json

    conn = sqlite3.connect(workspace / "data" / "dmn.sqlite")
    conn.execute(
        "INSERT OR REPLACE INTO dmn_meta(key, value, updated_at) VALUES (?, ?, 0)",
        ("profile_needs_recluster", json.dumps({"stale": True, "reason": "test"})),
    )
    conn.commit()
    conn.close()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "DMN_LLM_PROVIDER": "stub",
        "DMN_EMBEDDINGS": "hash",
    }
    r = subprocess.run(
        [sys.executable, str(REPO / "explore.py"), "--dry-run", "--iterations", "1"],
        cwd=workspace,
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    assert r.returncode == 1
    assert "mid-rebuild" in r.stdout
    # a successful prepare clears the mark
    r2 = run_prepare(workspace, "--interactive", stdin="bonsai\n\n\n\n\n\n")
    assert r2.returncode == 0
    r3 = subprocess.run(
        [sys.executable, str(REPO / "explore.py"), "--dry-run", "--iterations", "1"],
        cwd=workspace,
        text=True,
        capture_output=True,
        env=env,
        timeout=300,
    )
    assert r3.returncode == 0, r3.stdout + r3.stderr


def test_append_refuses_on_embedding_dim_mismatch(workspace: Path):
    env_openai_dim = {"DMN_EMBEDDINGS": "hash"}
    # Fake a stored 1536-dim profile by rewriting one embedding wider
    conn = sqlite3.connect(workspace / "data" / "dmn.sqlite")
    import json as _json

    wide = _json.dumps([0.01] * 1536)
    conn.execute("UPDATE interests SET embedding = ?", (wide,))
    conn.execute("DELETE FROM dmn_meta WHERE key = 'embedding_model'")  # unstamped legacy
    conn.commit()
    conn.close()
    r = run_prepare(workspace, "--interactive", stdin="totally new thing\n\n\n\n\n\n")
    assert r.returncode == 1
    assert "dimension" in r.stdout
    assert counts(workspace)[0] == 12  # nothing was written
