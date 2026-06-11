"""Taste-profile portability: serialize / deserialize a profile.dmn.json across DMN installs.

Schema v1:
{
  "schema_version": 1,
  "exported_at": ISO-8601,
  "embedding_model": {"name": ..., "dim": ..., "fingerprint": "sha256:..."},
  "interests": [{"text", "source", "weight", "embedding", "ts"}],
  "clusters":  [{"label", "centroid", "size", "weight"}],
  "meta": {"dmn_version", "interest_count", "cluster_count", "anonymized"}
}
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from typing import Any

import numpy as np

from dmn import __version__, embeddings, store

SCHEMA_VERSION = 6
_SUPPORTED_SCHEMAS = {1, 2, 3, 4, 5, 6}


def _embedding_model_name() -> str:
    """Return the canonical name of the embedding model actually in use."""
    backend = embeddings.effective_backend()
    if backend == "openai":
        return "openai/text-embedding-3-small"
    if backend == "hash":
        return "dmn/hash-fallback"
    return "sentence-transformers/all-MiniLM-L6-v2"


def _embedding_dim() -> int:
    """Return the expected embedding dimensionality for the current model."""
    name = _embedding_model_name()
    if "text-embedding-3-small" in name:
        return 1536
    return 384


EMBEDDING_MODEL_META_KEY = "embedding_model"


def stamp_profile_embedding(conn: sqlite3.Connection) -> None:
    """Record in the profile DB which embedding model produced its vectors."""
    store.set_meta(conn, EMBEDDING_MODEL_META_KEY, _embedding_model_name())


def profile_embedding_mismatch(conn: sqlite3.Connection) -> str | None:
    """Error text if the profile's vectors came from a different embedding backend.

    Returns None when the backends match or the profile predates stamping.
    Mixing backends silently corrupts every distance the dopamine score relies on,
    so callers should refuse to wander on a mismatch.
    """
    stored = store.get_meta(conn, EMBEDDING_MODEL_META_KEY)
    current = _embedding_model_name()
    if not stored or stored == current:
        return None
    if current == "dmn/hash-fallback" and "sentence-transformers" in stored:
        hint = "Run `uv sync --extra embeddings` to restore it, or rebuild with `uv run prepare.py`."
    else:
        hint = "Rebuild the profile with `uv run prepare.py`, or switch DMN_EMBEDDINGS back."
    return (
        f"This profile was embedded with {stored}, but the current backend is {current}. "
        f"Taste scores would be meaningless. {hint}"
    )


def profile_stale_reason(conn: sqlite3.Connection) -> str | None:
    """Reason text when a prepare rebuild was interrupted mid-flight, else None.

    prepare marks the profile stale before mutating an existing one and clears the
    mark only after clusters persist — wandering a half-rebuilt profile scores
    against clusters that no longer match the interests.
    """
    meta = store.get_meta(conn, "profile_needs_recluster")
    if isinstance(meta, dict) and meta.get("stale"):
        return (
            "This profile is mid-rebuild (a prepare run was interrupted: "
            f"{meta.get('reason') or 'unknown'}). Re-run `uv run prepare.py` to finish."
        )
    return None


def compute_embedding_fingerprint(model_name: str | None = None) -> str:
    """Stable per-model fingerprint used to gate import/export compatibility."""
    name = model_name or _embedding_model_name()
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def serialize_profile(
    conn: sqlite3.Connection, anonymize: bool = False
) -> dict[str, Any]:
    """Snapshot the SQLite profile to a JSON-safe dict (schema v1)."""
    interests = store.list_interests(conn)
    clusters = store.list_clusters(conn)
    model_name = _embedding_model_name()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": dt.datetime.utcnow().isoformat() + "Z",
        "embedding_model": {
            "name": model_name,
            "dim": _embedding_dim(),
            "fingerprint": compute_embedding_fingerprint(model_name),
        },
        "interests": [
            {
                "text": "" if anonymize else (i.get("text") or ""),
                "source": i.get("source") or "",
                "weight": float(i.get("weight") or 1.0),
                "embedding": _vec_to_list(i.get("embedding")),
                "ts": _iso_from_ts(i.get("timestamp")),
                "last_seen": _iso_from_ts(i.get("last_seen")),
                "tags": list(i.get("tags") or []),
            }
            for i in interests
        ],
        "clusters": [
            {
                "label": c.get("label") or "",
                "centroid": _vec_to_list(c.get("centroid")),
                "size": int(c.get("n_members") or 0),
                "weight": 1.0,
                "meta": c.get("meta") or None,
                "medoid_interest_id": c.get("medoid_interest_id"),
                "cluster_method": c.get("cluster_method"),
                "is_noise": bool(c.get("is_noise")),
            }
            for c in clusters
        ],
        "meta": {
            "dmn_version": __version__,
            "interest_count": len(interests),
            "cluster_count": len(clusters),
            "anonymized": bool(anonymize),
        },
    }
    return payload


def deserialize_profile(
    payload: dict[str, Any],
    conn: sqlite3.Connection,
    mode: str = "replace",
) -> dict[str, int]:
    """Load a payload into the local SQLite. mode: 'replace' or 'append'.

    Refuses if the payload's embedding fingerprint doesn't match the locally configured
    model (re-embedding across models is a v0.2 item).
    """
    incoming_schema = payload.get("schema_version")
    if incoming_schema not in _SUPPORTED_SCHEMAS:
        raise ValueError(
            f"unsupported schema_version: {incoming_schema}; this DMN supports {sorted(_SUPPORTED_SCHEMAS)}"
        )
    payload_fp = (payload.get("embedding_model") or {}).get("fingerprint") or ""
    local_fp = compute_embedding_fingerprint()
    if payload_fp != local_fp:
        raise ValueError(
            f"embedding fingerprint mismatch: payload={payload_fp} local={local_fp} "
            f"({_embedding_model_name()}). Re-export the profile from the source install, "
            "or match its embedding backend here first. Note: exports made before v0.5.1 "
            "from installs without sentence-transformers were mislabeled as all-MiniLM-L6-v2 "
            "even though their vectors were hash-derived. "
            "Re-embedding-on-import lands in v0.2."
        )
    if mode not in ("replace", "append"):
        raise ValueError(f"invalid mode: {mode}; expected replace or append")

    if mode == "replace":
        conn.execute("DELETE FROM interests")
        conn.execute("DELETE FROM clusters")
        conn.commit()

    n_int = 0
    for item in payload.get("interests", []) or []:
        emb_vec = _list_to_vec(item.get("embedding"))
        tags = item.get("tags") or None
        store.add_interest(
            conn,
            item.get("text") or "",
            item.get("source") or "imported",
            float(item.get("weight") or 1.0),
            emb_vec,
            tags=tags,
            last_seen=_ts_from_iso(item.get("last_seen") or item.get("ts")),
        )
        n_int += 1

    n_clu = 0
    incoming_clusters = payload.get("clusters") or []
    if mode == "replace" and incoming_clusters:
        centroids = np.stack(
            [
                _list_to_vec(c.get("centroid"))
                for c in incoming_clusters
                if c.get("centroid") is not None
            ]
        )
        labels = [
            c.get("label") or f"cluster-{i}" for i, c in enumerate(incoming_clusters)
        ]
        meta_per_cluster = [c.get("meta") for c in incoming_clusters]
        store.replace_clusters(
            conn,
            centroids,
            labels,
            meta_per_cluster=meta_per_cluster,
            # Restore noise flags: without them an imported catch-all bucket
            # becomes a normal cluster and the seed sampler stops skipping it.
            is_noise_flags=[bool(c.get("is_noise")) for c in incoming_clusters],
        )
        n_clu = len(incoming_clusters)
    if mode == "append":
        store.mark_profile_stale(
            conn,
            "profile append imported interests; rerun `uv run prepare.py --relabel-only` "
            "or rebuild clusters with prepare.py",
        )
    return {"interests_loaded": n_int, "clusters_loaded": n_clu}


def union_profiles(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Naive union of two profile payloads. v0.1.1 stub for `merge` — see TODO below."""
    if a.get("embedding_model", {}).get("fingerprint") != b.get(
        "embedding_model", {}
    ).get("fingerprint"):
        raise ValueError(
            "cannot union profiles built from different embedding models"
        )
    out = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": dt.datetime.utcnow().isoformat() + "Z",
        "embedding_model": a["embedding_model"],
        "interests": (a.get("interests") or []) + (b.get("interests") or []),
        # TODO(v0.2): real blended-clustering math goes here. We need to:
        #   1. concat embeddings from both profiles
        #   2. re-cluster (or do weighted Voronoi merge) honoring the --blend ratio
        #   3. assign labels via centroid-nearest-interest
        # For v0.1.1 we just union the cluster lists; the result is *not* a coherent merge.
        "clusters": (a.get("clusters") or []) + (b.get("clusters") or []),
        "meta": {
            "dmn_version": __version__,
            "interest_count": len(a.get("interests") or [])
            + len(b.get("interests") or []),
            "cluster_count": len(a.get("clusters") or [])
            + len(b.get("clusters") or []),
            "anonymized": False,
            "clusters_coherent": False,
            "needs_recluster": True,
            "merged_from": [
                a.get("meta", {}).get("dmn_version", "?"),
                b.get("meta", {}).get("dmn_version", "?"),
            ],
        },
    }
    return out


def _vec_to_list(v):
    """Convert a numpy vector to a JSON-safe list of floats; pass through None."""
    if v is None:
        return None
    return [float(x) for x in np.asarray(v).ravel().tolist()]


def _list_to_vec(lst):
    """Convert a list of floats back to a float32 numpy vector; pass through None."""
    if not lst:
        return None
    return np.asarray(lst, dtype=np.float32)


def _iso_from_ts(ts) -> str:
    """Turn a unix timestamp (or None) into an ISO-8601 string in UTC."""
    if not ts:
        return ""
    try:
        return dt.datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
    except Exception:
        return ""


def _ts_from_iso(value) -> float | None:
    """Parse an ISO timestamp from profile JSON into unix seconds."""
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None
