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
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recent_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding TEXT,
    created_at REAL NOT NULL
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


SCHEMA_VERSION = 5


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent on-startup migrations.

    v2: clusters.meta + interests.tags
    v3: journal tree-search columns (parent_id, mutation, depth, status, subtree_score)
    v4: journal entity-tagging columns (entities, rabbit_holes — both JSON TEXT)
    v5: journal activity columns (activity, artifact_paths, execution_result)
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
    ]:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (v INTEGER PRIMARY KEY)")
    conn.execute("INSERT OR IGNORE INTO schema_version(v) VALUES (?)", (SCHEMA_VERSION,))
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
) -> int:
    """Insert an interest row; returns the new row id."""
    cur = conn.execute(
        "INSERT INTO interests(text, source, timestamp, weight, embedding, tags) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            text,
            source,
            time.time(),
            weight,
            _vec_to_json(embedding),
            json.dumps(tags) if tags else None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_interests(conn: sqlite3.Connection) -> list[dict]:
    """Return all interests as a list of dicts (embedding as np.ndarray or None; tags as list)."""
    rows = conn.execute(
        "SELECT id, text, source, timestamp, weight, embedding, tags FROM interests"
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
) -> None:
    """Replace the entire clusters table with the given centroids, labels, and optional meta."""
    conn.execute("DELETE FROM clusters")
    now = time.time()
    for i, c in enumerate(centroids):
        label = labels[i] if i < len(labels) else f"cluster-{i}"
        meta_json = None
        if meta_per_cluster and i < len(meta_per_cluster) and meta_per_cluster[i]:
            meta_json = json.dumps(meta_per_cluster[i])
        conn.execute(
            "INSERT INTO clusters(id, centroid, label, n_members, updated_at, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _vec_to_json(np.asarray(c)), label, 0, now, meta_json),
        )
    conn.commit()


def list_clusters(conn: sqlite3.Connection) -> list[dict]:
    """Return all clusters as a list of dicts (includes parsed `meta` if present)."""
    rows = conn.execute(
        "SELECT id, centroid, label, n_members, meta FROM clusters"
    ).fetchall()
    return [
        {
            "id": r[0],
            "centroid": _vec_from_json(r[1]),
            "label": r[2],
            "n_members": r[3],
            "meta": json.loads(r[4]) if r[4] else None,
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
    "subtree_score, entities, rabbit_holes, activity, artifact_paths, execution_result"
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
    cur = conn.execute(
        "INSERT INTO journal(seed, seed_source, tools, dopamine_total, "
        "dopamine_breakdown, path, embedding, created_at, "
        "parent_id, mutation, depth, status, subtree_score, entities, rabbit_holes, "
        "activity, artifact_paths, execution_result) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        ),
    )
    conn.commit()
    return cur.lastrowid


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
