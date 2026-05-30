"""Programmatic wander runner for CLI wrappers and remote API hosts."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dmn import store
from dmn.wander_config import WanderConfig, WanderResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WANDER_SCRIPT = _REPO_ROOT / "wander.py"


@contextmanager
def _workdir(path: Path) -> Iterator[Path]:
    """Prepare an isolated workspace and chdir into it for the wander subprocess."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "journal").mkdir(parents=True, exist_ok=True)
    (path / "data" / "artifacts").mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def _apply_llm_env(config: WanderConfig) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for key, val in (
        ("DMN_LLM_PROVIDER", config.llm_provider),
        ("DMN_LLM_MODEL", config.llm_model),
        ("DMN_LLM_BASE_URL", config.llm_base_url),
    ):
        if val is not None:
            prev[key] = os.environ.get(key)
            os.environ[key] = val
    return prev


def _restore_llm_env(prev: dict[str, str | None]) -> None:
    for key, val in prev.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def build_wander_argv(config: WanderConfig) -> list[str]:
    """Translate a WanderConfig into `wander.py` CLI arguments."""
    argv = [sys.executable, str(_WANDER_SCRIPT)]
    argv += ["--minutes", str(config.minutes)]
    if config.iterations is not None:
        argv += ["--iterations", str(config.iterations)]
    argv += [
        "--max-depth",
        str(config.max_depth),
        "--children-per-expansion",
        str(config.children_per_expansion),
        "--root-count",
        str(config.root_count),
        "--restart-prob",
        str(config.restart_prob),
        "--margin",
        str(config.margin),
        "--min-absolute",
        str(config.min_absolute),
        "--similarity-threshold",
        str(config.similarity_threshold),
        "--activities",
        config.activities,
        "--sandbox",
        config.sandbox,
        "--modalities",
        config.modalities,
        "--beam-width",
        str(config.beam_width),
        "--patience",
        str(config.patience),
        "--min-improvement",
        str(config.min_improvement),
        "--explore",
        str(config.explore),
        "--code-budget",
        config.code_budget,
    ]
    if config.seed_text:
        argv += ["--seed", config.seed_text]
    if config.activity_mix:
        argv += ["--activity-mix", config.activity_mix]
    if config.execute:
        argv.append("--execute")
    if config.no_execute:
        argv.append("--no-execute")
    if config.dry_run:
        argv.append("--dry-run")
    if config.generate:
        argv.append("--generate")
    if config.resume:
        argv.append("--resume")
    if config.until_dopamine is not None:
        argv += ["--until-dopamine", str(config.until_dopamine)]
    if not config.ground:
        argv.append("--no-ground")
    if not config.report:
        argv.append("--no-report")
    if config.verbose:
        argv.append("--verbose")
    return argv


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    fm: dict[str, Any] = {}
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        try:
            fm[key] = json.loads(val)
        except json.JSONDecodeError:
            fm[key] = val.strip('"')
    return fm, parts[2].lstrip("\n")


def _collect_files(work_dir: Path) -> dict[str, str]:
    """Map repo-relative paths to absolute paths for every generated artifact."""
    files: dict[str, str] = {}
    for sub in ("journal", "data"):
        root = work_dir / sub
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(work_dir))
                files[rel] = str(p)
    return files


def _latest_run(conn: sqlite3.Connection) -> tuple[str, dict[str, Any]]:
    row = conn.execute(
        "SELECT id, started_at, finished_at, best_score, brief_count, notes "
        "FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("wander finished but no run row was recorded")
    return row[0], {
        "started_at": row[1],
        "finished_at": row[2],
        "best_score": row[3],
        "brief_count": row[4],
        "notes": row[5],
    }


def collect_wander_result(work_dir: Path, run_id: str | None = None) -> WanderResult:
    """Read wander outputs from an isolated workspace."""
    db_path = work_dir / "data" / "dmn.sqlite"
    conn = store.connect(db_path)
    try:
        if run_id is None:
            run_id, run_meta = _latest_run(conn)
        else:
            row = conn.execute(
                "SELECT id, started_at, finished_at, best_score, brief_count, notes "
                "FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                raise RuntimeError(f"run_id not found: {run_id}")
            run_meta = {
                "started_at": row[1],
                "finished_at": row[2],
                "best_score": row[3],
                "brief_count": row[4],
                "notes": row[5],
            }
            run_id = row[0]

        briefs = store.list_journal_by_run(conn, run_id, limit=1000)
        started = float(run_meta.get("started_at") or 0.0)
        finished = float(run_meta.get("finished_at") or time.time())
        notes = str(run_meta.get("notes") or "")
        patience_triggered = "patience_triggered=True" in notes
        beam_pruned = 0
        m = re.search(r"beam_pruned=(\d+)", notes)
        if m:
            beam_pruned = int(m.group(1))

        pruned_count = sum(1 for b in briefs if b.get("status") == "pruned")
        leaf_count = sum(1 for b in briefs if b.get("status") == "leaf")
        open_count = sum(1 for b in briefs if b.get("status") == "open")

        report_path = None
        for candidate in (
            work_dir / "journal" / f"report-{run_id}.md",
            work_dir / "journal" / "report.md",
        ):
            if candidate.exists():
                report_path = str(candidate.relative_to(work_dir))
                break

        tree_path = None
        tree_md = work_dir / "journal" / "tree.md"
        if tree_md.exists():
            tree_path = str(tree_md.relative_to(work_dir))

        # Enrich brief rows with parsed markdown bodies where available.
        enriched: list[dict[str, Any]] = []
        for b in briefs:
            row = dict(b)
            row.pop("embedding", None)
            rel_path = row.get("path")
            if rel_path:
                brief_file = work_dir / rel_path
                if brief_file.exists():
                    fm, body = _parse_frontmatter(brief_file.read_text(encoding="utf-8"))
                    row["frontmatter"] = fm
                    row["body_md"] = body
            enriched.append(row)

        return WanderResult(
            run_id=run_id,
            work_dir=str(work_dir),
            brief_count=int(run_meta.get("brief_count") or len(briefs)),
            best_score=float(run_meta.get("best_score") or 0.0),
            pruned_count=pruned_count,
            leaf_count=leaf_count,
            open_count=open_count,
            duration_seconds=max(0.0, finished - started),
            patience_triggered=patience_triggered,
            beam_pruned=beam_pruned,
            briefs=enriched,
            report_path=report_path,
            tree_path=tree_path,
            files=_collect_files(work_dir),
        )
    finally:
        conn.close()


def run_wander(
    config: WanderConfig,
    *,
    profile_db: Path,
    work_dir: Path,
    timeout_s: float | None = None,
) -> WanderResult:
    """Copy a profile SQLite into `work_dir`, run wander.py, return structured results."""
    profile_db = profile_db.resolve()
    if not profile_db.is_file():
        raise FileNotFoundError(f"profile database not found: {profile_db}")

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile_db, work_dir / "data" / "dmn.sqlite")

    argv = build_wander_argv(config)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT)
    prev_llm = _apply_llm_env(config)
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(work_dir),
            env=env,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"wander.py exited with code {proc.returncode}")
        result = collect_wander_result(work_dir)
        if result.duration_seconds <= 0:
            result.duration_seconds = time.time() - started
        return result
    finally:
        _restore_llm_env(prev_llm)
