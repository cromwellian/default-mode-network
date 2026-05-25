"""SQLite storage for taste profile, clusters, and journal index. Vectors are stored as JSON."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

DB_PATH = Path("data/dmn.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp REAL NOT NULL,
    weight REAL DEFAULT 1.0,
    embedding TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    centroid TEXT NOT NULL,
    label TEXT,
    n_members INTEGER DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed TEXT,
    seed_source TEXT,
    tools TEXT,
    dopamine_total REAL,
    dopamine_breakdown TEXT,
    path TEXT NOT NULL,
    embedding TEXT,
    created_at REAL NOT NULL,
    run_id TEXT,
    visit_count INTEGER DEFAULT 0,
    expanded_count INTEGER DEFAULT 0,
    last_improvement REAL,
    activity_budget TEXT,
    cost_seconds REAL,
    user_rating REAL,
    user_feedback TEXT,
    reviewed_at REAL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at REAL,
    finished_at REAL,
    command TEXT,
    notes TEXT,
    best_score REAL,
    brief_count INTEGER
);

CREATE TABLE IF NOT EXISTS recent_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dmn_meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL NOT NULL
);
"""


def _vec_to_json(v: Optional[np.ndarray]) -> Optional[str]:
    """Serialize a numpy vector to a JSON list string, or None."""
    if v is None:
        return None
    return json.dumps([float(x) for x in np.asarray(v).ravel().tolist()])


def _vec_from_json(s: Optional[str]) -> Optional[np.ndarray]:
    """Deserialize a JSON list string into a float32 numpy vector, or None."""
    if not s:
        return None
    return np.asarray(json.loads(s), dtype=np.float32)


SCHEMA_VERSION = 9


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent on-startup migrations.

    v2: clusters.meta + interests.tags
    v3: journal tree-search columns (parent_id, mutation, depth, status, subtree_score)
    v4: journal entity-tagging columns (entities, rabbit_holes — both JSON TEXT)
    v5: journal activity columns (activity, artifact_paths, execution_result)
    v6: HDBSCAN medoids, cluster_method, is_noise, interests.last_seen
    v7: run/session identity + journal metrics
    v8: fulfillment + grounding references persisted on the journal row
    v9: user ratings/feedback + explicit metadata table
    """
    for sql in [
        # v2
        "ALTER TABLE clusters ADD COLUMN meta TEXT",
        "ALTER TABLE interests ADD COLUMN tags TEXT",
        # v3 — best-first tree-search wander mode
        "ALTER TABLE journal ADD COLUMN parent_id INTEGER REFERENCES journal(id)",
        "ALTER TABLE journal ADD COLUMN mutation TEXT",
        "ALTER TABLE journal ADD COLUMN depth INTEGER DEFAULT 0",
        "ALTER TABLE journal ADD COLUMN status TEXT DEFAULT 'open'",
        "ALTER TABLE journal ADD COLUMN subtree_score REAL",
        # v4 — entity tagging on brief synthesis (powers mutate_deepen)
        "ALTER TABLE journal ADD COLUMN entities TEXT",
        "ALTER TABLE journal ADD COLUMN rabbit_holes TEXT",
        # v5 — pluggable wander activities
        "ALTER TABLE journal ADD COLUMN activity TEXT",
        "ALTER TABLE journal ADD COLUMN artifact_paths TEXT",
        "ALTER TABLE journal ADD COLUMN execution_result TEXT",
        # v6 — HDBSCAN clustering metadata
        "ALTER TABLE clusters ADD COLUMN medoid_interest_id INTEGER",
        "ALTER TABLE clusters ADD COLUMN cluster_method TEXT",
        "ALTER TABLE clusters ADD COLUMN is_noise INTEGER DEFAULT 0",
        "ALTER TABLE interests ADD COLUMN last_seen REAL",
        # v7 — run/session identity and journal metrics
        "ALTER TABLE journal ADD COLUMN run_id TEXT",
        "ALTER TABLE journal ADD COLUMN visit_count INTEGER DEFAULT 0",
        "ALTER TABLE journal ADD COLUMN expanded_count INTEGER DEFAULT 0",
        "ALTER TABLE journal ADD COLUMN last_improvement REAL",
        "ALTER TABLE journal ADD COLUMN activity_budget TEXT",
        "ALTER TABLE journal ADD COLUMN cost_seconds REAL",
        # v8 — reward + grounding provenance
        "ALTER TABLE journal ADD COLUMN fulfillment REAL",
        "ALTER TABLE journal ADD COLUMN fulfillment_breakdown TEXT",
        "ALTER TABLE journal ADD COLUMN grounding TEXT",
        # v9 — explicit user feedback for reward calibration
        "ALTER TABLE journal ADD COLUMN user_rating REAL",
        "ALTER TABLE journal ADD COLUMN user_feedback TEXT",
        "ALTER TABLE journal ADD COLUMN reviewed_at REAL",
    ]:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (v INTEGER PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dmn_meta ("
        "key TEXT PRIMARY KEY, value TEXT, updated_at REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "id TEXT PRIMARY KEY, started_at REAL, finished_at REAL, command TEXT, "
        "notes TEXT, best_score REAL, brief_count INTEGER)"
    )
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version(v) VALUES (?)", (SCHEMA_VERSION,))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open (and initialize + migrate) a SQLite connection at the given path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def add_interest(
    conn: sqlite3.Connection,
    text: str,
    source: str,
    weight: float = 1.0,
    embedding: Optional[np.ndarray] = None,
    tags: Optional[list[str]] = None,
    last_seen: Optional[float] = None,
) -> int:
    """Insert an interest row; returns the new row id."""
    cur = conn.execute(
        "INSERT INTO interests(text, source, timestamp, weight, embedding, tags, last_seen) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            text,
            source,
            time.time(),
            weight,
            _vec_to_json(embedding),
            json.dumps(tags) if tags else None,
            last_seen,
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_interests(conn: sqlite3.Connection) -> list[dict]:
    """Return all interests as a list of dicts (embedding as np.ndarray or None; tags as list)."""
    rows = conn.execute(
        "SELECT id, text, source, timestamp, weight, embedding, tags, last_seen "
        "FROM interests"
    ).fetchall()
    return [
        {
            "id": r[0],
            "text": r[1],
            "source": r[2],
            "timestamp": r[3],
            "weight": r[4],
            "embedding": _vec_from_json(r[5]),
            "tags": json.loads(r[6]) if r[6] else [],
            "last_seen": r[7],
        }
        for r in rows
    ]


def set_interest_tags(
    conn: sqlite3.Connection, interest_id: int, tags: Optional[list[str]]
) -> None:
    """Overwrite an interest's `tags` column (JSON-encoded list)."""
    conn.execute(
        "UPDATE interests SET tags = ? WHERE id = ?",
        (json.dumps(tags) if tags else None, interest_id),
    )
    conn.commit()


def replace_clusters(
    conn: sqlite3.Connection,
    centroids: np.ndarray,
    labels: list[str],
    meta_per_cluster: Optional[list[dict]] = None,
    n_members: Optional[list[int]] = None,
    medoid_interest_ids: Optional[list[int]] = None,
    cluster_method: str = "hdbscan",
    is_noise_flags: Optional[list[bool]] = None,
) -> None:
    """Replace the entire clusters table with the given centroids, labels, and optional meta."""
    conn.execute("DELETE FROM clusters")
    now = time.time()
    for i, c in enumerate(centroids):
        label = labels[i] if i < len(labels) else f"cluster-{i}"
        meta_json = None
        if meta_per_cluster and i < len(meta_per_cluster) and meta_per_cluster[i]:
            meta_json = json.dumps(meta_per_cluster[i])
        nm = int(n_members[i]) if n_members and i < len(n_members) else 0
        medoid_id = (
            medoid_interest_ids[i]
            if medoid_interest_ids and i < len(medoid_interest_ids)
            else None
        )
        is_noise = (
            int(bool(is_noise_flags[i]))
            if is_noise_flags and i < len(is_noise_flags)
            else 0
        )
        conn.execute(
            "INSERT INTO clusters(id, centroid, label, n_members, updated_at, meta, "
            "medoid_interest_id, cluster_method, is_noise) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                i,
                _vec_to_json(np.asarray(c)),
                label,
                nm,
                now,
                meta_json,
                medoid_id,
                cluster_method,
                is_noise,
            ),
        )
    conn.commit()


def list_clusters(conn: sqlite3.Connection) -> list[dict]:
    """Return all clusters as a list of dicts (includes parsed `meta` if present)."""
    rows = conn.execute(
        "SELECT id, centroid, label, n_members, meta, medoid_interest_id, "
        "cluster_method, is_noise FROM clusters"
    ).fetchall()
    return [
        {
            "id": r[0],
            "centroid": _vec_from_json(r[1]),
            "label": r[2],
            "n_members": r[3],
            "meta": json.loads(r[4]) if r[4] else None,
            "medoid_interest_id": r[5],
            "cluster_method": r[6],
            "is_noise": bool(r[7]) if r[7] is not None else False,
        }
        for r in rows
    ]


def update_cluster(
    conn: sqlite3.Connection, cid: int, new_centroid: np.ndarray
) -> None:
    """Overwrite a single cluster's centroid (used for online taste-profile nudges)."""
    conn.execute(
        "UPDATE clusters SET centroid = ?, updated_at = ? WHERE id = ?",
        (_vec_to_json(new_centroid), time.time(), cid),
    )
    conn.commit()


def update_cluster_meta(
    conn: sqlite3.Connection, cid: int, meta: Optional[dict]
) -> None:
    """Overwrite a single cluster's `meta` JSON (theme + subtopics + rationale)."""
    conn.execute(
        "UPDATE clusters SET meta = ?, updated_at = ? WHERE id = ?",
        (json.dumps(meta) if meta else None, time.time(), cid),
    )
    conn.commit()


def update_cluster_label(
    conn: sqlite3.Connection, cid: int, label: str
) -> None:
    """Overwrite a single cluster's human-readable `label`."""
    conn.execute(
        "UPDATE clusters SET label = ?, updated_at = ? WHERE id = ?",
        (label, time.time(), cid),
    )
    conn.commit()


_JOURNAL_COLUMNS = (
    "id, seed, seed_source, tools, dopamine_total, dopamine_breakdown, "
    "path, embedding, created_at, parent_id, mutation, depth, status, "
    "subtree_score, entities, rabbit_holes, activity, artifact_paths, execution_result, "
    "run_id, visit_count, expanded_count, last_improvement, activity_budget, cost_seconds, "
    "fulfillment, fulfillment_breakdown, grounding, user_rating, user_feedback, reviewed_at"
)


def _row_to_journal(row: tuple) -> dict:
    """Map a `_JOURNAL_COLUMNS`-shaped row tuple to a dict (with parsed JSON / vector fields)."""
    return {
        "id": row[0],
        "seed": row[1],
        "seed_source": row[2],
        "tools": row[3].split(",") if row[3] else [],
        "dopamine_total": row[4],
        "dopamine_breakdown": json.loads(row[5]) if row[5] else {},
        "path": row[6],
        "embedding": _vec_from_json(row[7]),
        "created_at": row[8],
        "parent_id": row[9],
        "mutation": row[10],
        "depth": int(row[11] or 0),
        "status": row[12] or "open",
        "subtree_score": row[13],
        "entities": json.loads(row[14]) if row[14] else [],
        "rabbit_holes": json.loads(row[15]) if row[15] else [],
        "activity": row[16],
        "artifact_paths": json.loads(row[17]) if row[17] else [],
        "execution_result": json.loads(row[18]) if row[18] else None,
        "run_id": row[19],
        "visit_count": int(row[20] or 0),
        "expanded_count": int(row[21] or 0),
        "last_improvement": row[22],
        "activity_budget": row[23],
        "cost_seconds": row[24],
        "fulfillment": row[25],
        "fulfillment_breakdown": json.loads(row[26]) if row[26] else {},
        "grounding": json.loads(row[27]) if row[27] else [],
        "user_rating": row[28],
        "user_feedback": row[29],
        "reviewed_at": row[30],
    }


def add_journal(
    conn: sqlite3.Connection,
    seed: str,
    seed_source: str,
    tools: list[str],
    dopamine: dict,
    path: str,
    embedding: Optional[np.ndarray] = None,
    parent_id: Optional[int] = None,
    mutation: Optional[str] = None,
    depth: int = 0,
    status: str = "open",
    entities: Optional[list] = None,
    rabbit_holes: Optional[list] = None,
    activity: Optional[str] = None,
    artifact_paths: Optional[list] = None,
    execution_result: Optional[dict] = None,
    run_id: Optional[str] = None,
    visit_count: int = 0,
    expanded_count: int = 0,
    last_improvement: Optional[float] = None,
    activity_budget: Optional[str] = None,
    cost_seconds: Optional[float] = None,
    fulfillment: Optional[float] = None,
    fulfillment_breakdown: Optional[dict] = None,
    grounding: Optional[list] = None,
) -> int:
    """Record a brief in the journal index; returns its row id.

    v0.2 adds tree-search fields (`parent_id`, `mutation`, `depth`, `status`); flat-mode
    callers who don't pass them get sensible defaults (`mutation=None`, `depth=0`,
    `status='open'`) — fully back-compat with v0.1.x callers.

    v0.2.1 adds `entities` and `rabbit_holes` (lists, default empty); these power
    `mutate_deepen` / `mutate_branch` on subsequent expansions.

    v0.3 adds `activity` ('research' | 'code_sketch' | ... ), `artifact_paths` (list of
    string disk paths produced by the activity), and `execution_result` (sandbox result
    dict, when --execute was on).
    """
    entities_json = json.dumps(entities) if entities else None
    rabbit_holes_json = json.dumps(rabbit_holes) if rabbit_holes else None
    artifact_paths_json = json.dumps(artifact_paths) if artifact_paths else None
    execution_json = json.dumps(execution_result) if execution_result else None
    fulfillment_breakdown_json = (
        json.dumps(fulfillment_breakdown) if fulfillment_breakdown else None
    )
    grounding_json = json.dumps(grounding) if grounding else None
    cur = conn.execute(
        "INSERT INTO journal(seed, seed_source, tools, dopamine_total, "
        "dopamine_breakdown, path, embedding, created_at, "
        "parent_id, mutation, depth, status, subtree_score, entities, rabbit_holes, "
        "activity, artifact_paths, execution_result, run_id, visit_count, expanded_count, "
        "last_improvement, activity_budget, cost_seconds, "
        "fulfillment, fulfillment_breakdown, grounding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            seed,
            seed_source,
            ",".join(tools),
            float(dopamine.get("total", 0.0)),
            json.dumps(dopamine),
            path,
            _vec_to_json(embedding),
            time.time(),
            parent_id,
            mutation,
            int(depth),
            status,
            float(dopamine.get("total", 0.0)),
            entities_json,
            rabbit_holes_json,
            activity,
            artifact_paths_json,
            execution_json,
            run_id,
            int(visit_count),
            int(expanded_count),
            last_improvement,
            activity_budget,
            cost_seconds,
            fulfillment,
            fulfillment_breakdown_json,
            grounding_json,
        ),
    )
    conn.commit()
    return cur.lastrowid


def start_run(conn: sqlite3.Connection, run_id: str, command: str = "", notes: str = "") -> None:
    """Create or refresh a run row for a wander/explore invocation."""
    conn.execute(
        "INSERT OR REPLACE INTO runs(id, started_at, finished_at, command, notes, best_score, brief_count) "
        "VALUES (?, ?, NULL, ?, ?, NULL, 0)",
        (run_id, time.time(), command, notes),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    best_score: Optional[float] = None,
    brief_count: int = 0,
    notes: str = "",
) -> None:
    """Mark a run complete and store summary metrics."""
    conn.execute(
        "UPDATE runs SET finished_at = ?, best_score = ?, brief_count = ?, "
        "notes = COALESCE(NULLIF(?, ''), notes) WHERE id = ?",
        (time.time(), best_score, int(brief_count), notes, run_id),
    )
    conn.commit()


def increment_journal_visit(conn: sqlite3.Connection, journal_id: int) -> None:
    """Increment visit_count for a journal node when it is selected from the frontier."""
    conn.execute(
        "UPDATE journal SET visit_count = COALESCE(visit_count, 0) + 1 WHERE id = ?",
        (journal_id,),
    )
    conn.commit()


def increment_journal_expanded(conn: sqlite3.Connection, journal_id: int) -> None:
    """Increment expanded_count for a journal node after child expansion is attempted."""
    conn.execute(
        "UPDATE journal SET expanded_count = COALESCE(expanded_count, 0) + 1 WHERE id = ?",
        (journal_id,),
    )
    conn.commit()


def update_journal_last_improvement(
    conn: sqlite3.Connection, journal_id: int, improvement: float
) -> None:
    """Record the score delta observed when this journal node was created."""
    conn.execute(
        "UPDATE journal SET last_improvement = ? WHERE id = ?",
        (float(improvement), journal_id),
    )
    conn.commit()


def list_journal_by_activity(
    conn: sqlite3.Connection, name: str, limit: int = 50
) -> list[dict]:
    """Return recent journal entries for a single activity, newest first."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE activity = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (name, limit),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def list_journal_by_run(
    conn: sqlite3.Connection, run_id: str, limit: int = 1000
) -> list[dict]:
    """Return journal entries for one run, newest first."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE run_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def get_entities(conn: sqlite3.Connection, journal_id: int) -> list[dict]:
    """Return the structured entities recorded for a journal row, or [] if none."""
    row = conn.execute(
        "SELECT entities FROM journal WHERE id = ?", (journal_id,)
    ).fetchone()
    if not row or not row[0]:
        return []
    try:
        data = json.loads(row[0])
    except Exception:
        return []
    return data if isinstance(data, list) else []


def list_journal(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Return recent journal entries (newest first), up to `limit`. Includes v0.2 fields."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def get_journal(conn: sqlite3.Connection, journal_id: int) -> Optional[dict]:
    """Fetch a single journal row by id, or None if it doesn't exist."""
    row = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE id = ?", (journal_id,)
    ).fetchone()
    return _row_to_journal(row) if row else None


def children_of(conn: sqlite3.Connection, journal_id: int) -> list[dict]:
    """Return the children of a journal row, ordered by id ascending."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE parent_id = ? ORDER BY id ASC",
        (journal_id,),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def list_open_frontier(conn: sqlite3.Connection) -> list[dict]:
    """Return all journal rows with status='open' (the live tree-search frontier)."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE status = 'open' "
        "ORDER BY (subtree_score IS NULL), subtree_score DESC, dopamine_total DESC"
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def update_journal_status(
    conn: sqlite3.Connection, journal_id: int, status: str
) -> None:
    """Set a journal row's `status` (typically 'open' / 'leaf' / 'pruned')."""
    conn.execute(
        "UPDATE journal SET status = ? WHERE id = ?", (status, journal_id)
    )
    conn.commit()


def update_journal_subtree_score(
    conn: sqlite3.Connection, journal_id: int, score: float
) -> None:
    """Overwrite a journal row's backprop'd `subtree_score`."""
    conn.execute(
        "UPDATE journal SET subtree_score = ? WHERE id = ?",
        (float(score), journal_id),
    )
    conn.commit()


def add_finding(
    conn: sqlite3.Connection,
    text: str,
    embedding: Optional[np.ndarray] = None,
    keep_last: int = 200,
) -> None:
    """Append a recent finding for novelty/anti-repetition; keep only the most recent `keep_last`."""
    conn.execute(
        "INSERT INTO recent_findings(text, embedding, created_at) VALUES (?, ?, ?)",
        (text, _vec_to_json(embedding), time.time()),
    )
    conn.execute(
        "DELETE FROM recent_findings WHERE id NOT IN "
        "(SELECT id FROM recent_findings ORDER BY id DESC LIMIT ?)",
        (keep_last,),
    )
    conn.commit()


def list_recent_findings(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Return up to `limit` most-recent findings (newest first)."""
    rows = conn.execute(
        "SELECT text, embedding FROM recent_findings ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"text": r[0], "embedding": _vec_from_json(r[1])} for r in rows]


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    """Set a small repository/profile metadata value as JSON."""
    conn.execute(
        "INSERT OR REPLACE INTO dmn_meta(key, value, updated_at) VALUES (?, ?, ?)",
        (str(key), json.dumps(value), time.time()),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    """Read a metadata value written by `set_meta`, returning default when absent."""
    row = conn.execute("SELECT value FROM dmn_meta WHERE key = ?", (str(key),)).fetchone()
    if not row or row[0] is None:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return default


def mark_profile_stale(conn: sqlite3.Connection, reason: str) -> None:
    """Mark cluster labels/centroids as stale after import/merge-style operations."""
    set_meta(
        conn,
        "profile_needs_recluster",
        {"stale": True, "reason": reason, "marked_at": time.time()},
    )


def clear_profile_stale(conn: sqlite3.Connection) -> None:
    """Clear the stale-profile marker after clustering succeeds."""
    set_meta(conn, "profile_needs_recluster", {"stale": False, "reason": "", "marked_at": time.time()})


def update_journal_rating(
    conn: sqlite3.Connection,
    journal_id: int,
    rating: Optional[float],
    feedback: Optional[str] = None,
) -> None:
    """Attach a user delight rating and optional feedback to a journal row."""
    if rating is None:
        rating_value = None
    else:
        rating_value = max(1.0, min(5.0, float(rating)))
    conn.execute(
        "UPDATE journal SET user_rating = ?, user_feedback = ?, reviewed_at = ? WHERE id = ?",
        (rating_value, feedback, time.time(), int(journal_id)),
    )
    conn.commit()


def list_rated_journal(conn: sqlite3.Connection, limit: int = 500) -> list[dict]:
    """Return recent journal entries with a user rating, newest reviews first."""
    rows = conn.execute(
        f"SELECT {_JOURNAL_COLUMNS} FROM journal WHERE user_rating IS NOT NULL "
        "ORDER BY reviewed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]
