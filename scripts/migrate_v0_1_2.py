"""One-shot migration from v0.1.0 / v0.1.1 SQLite profiles to v0.1.2.

Run this once after upgrading to v0.1.2 if your existing profile contains noisy
browser-history rows (homepage visits, login pages, generic-domain titles) and stale
URL-flavored cluster labels:

    uv run scripts/migrate_v0_1_2.py [--dry-run]

What it does:

  1. Re-runs the v0.1.2 browser-importer filter (`_clean_history_rows`) against existing
     `browser:*` interest rows; deletes rows that fail the new filter.
  2. Re-clusters the remaining interests in embedding space.
  3. Re-labels each cluster via `dmn.labeling.synthesize_label` (LLM if configured, else
     stub-based word-frequency heuristic).

Idempotent: running it again is a no-op once the noise has been cleaned out and labels
have been refreshed.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Allow `python scripts/migrate_v0_1_2.py` from anywhere by injecting the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import embeddings as emb
from dmn import labeling, store, taste
from dmn.importers.browser import _clean_history_rows
from dmn.llm import get_llm

console = Console()
_TITLE_URL_RE = re.compile(r"^(.*) \((https?://[^)]+)\)$")


def main(dry_run: bool = False) -> None:
    """Run the v0.1.2 migration in-place against `data/dmn.sqlite`."""
    conn = store.connect()  # _migrate() runs automatically on connect
    interests = store.list_interests(conn)
    total_before = len(interests)
    console.print(f"Profile contains [bold]{total_before}[/] interest rows before migration.")

    # 1) Re-filter browser:* rows.
    browser_rows = []
    for row in interests:
        src = row.get("source") or ""
        if not src.startswith("browser:"):
            continue
        m = _TITLE_URL_RE.match(row.get("text", "") or "")
        title, url = (m.group(1), m.group(2)) if m else (row.get("text", "") or "", "")
        browser_rows.append({"id": row["id"], "title": title.strip(), "url": url})

    raw_for_filter = [{"title": r["title"], "url": r["url"]} for r in browser_rows]
    kept = _clean_history_rows(raw_for_filter)
    kept_keys = {(r["title"], r["url"]) for r in kept}
    drop_ids = [
        r["id"] for r in browser_rows if (r["title"], r["url"]) not in kept_keys
    ]
    console.print(
        f"Browser-row filter: keeping [bold]{len(browser_rows) - len(drop_ids)}[/], "
        f"dropping [bold]{len(drop_ids)}[/] noisy rows."
    )
    if not dry_run and drop_ids:
        for chunk_start in range(0, len(drop_ids), 500):
            chunk = drop_ids[chunk_start : chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            conn.execute(
                f"DELETE FROM interests WHERE id IN ({placeholders})", chunk
            )
        conn.commit()

    # 2) Re-cluster the (now cleaner) profile.
    interests = store.list_interests(conn)
    if not interests:
        console.print("[red]No interests left. Aborting.[/]")
        return

    vectors_list: list[np.ndarray] = []
    texts: list[str] = []
    for i in interests:
        if i["embedding"] is not None:
            vectors_list.append(i["embedding"])
        else:
            vectors_list.append(emb.embed([i["text"]])[0])
        texts.append(i["text"] or "")
    vectors = np.stack(vectors_list).astype(np.float32)

    cr = taste.cluster(vectors, method="kmeans")
    centroids, labels = cr.centroids, cr.labels
    console.print(f"Re-clustered into [bold]{len(centroids)}[/] cluster(s).")

    # 3) Re-label.
    llm = get_llm(dry_run=dry_run)
    console.print(f"Synthesizing labels via [bold]{llm.name}[/] LLM…")
    cluster_themes: list[str] = []
    cluster_metas: list[dict] = []
    for ci in range(len(centroids)):
        idx = [j for j, lab in enumerate(labels) if int(lab) == ci]
        if not idx:
            cluster_themes.append(f"cluster-{ci}")
            cluster_metas.append({})
            continue
        ranked = sorted(
            idx, key=lambda j: float(np.linalg.norm(vectors[j] - centroids[ci]))
        )
        seen: set[str] = set()
        chosen: list[str] = []
        for j in ranked:
            t = (texts[j] or "").strip()
            key = t.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append(t[:200])
            if len(chosen) >= 12:
                break
        cl = labeling.synthesize_label(chosen, llm)
        cluster_themes.append(cl.theme or f"cluster-{ci}")
        cluster_metas.append(cl.to_meta())
        console.print(f"  [bold cyan]{ci}.[/] {cl.theme}")
        if cl.subtopics:
            console.print(f"     subtopics: {cl.subtopics}")

    if not dry_run:
        store.replace_clusters(
            conn, centroids, cluster_themes, meta_per_cluster=cluster_metas
        )
        console.print("[green]Migration complete.[/]")
    else:
        console.print("[yellow]--dry-run set: no changes persisted.[/]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the SQLite or calling a real LLM.",
    )
    args = ap.parse_args()
    main(dry_run=args.dry_run)
