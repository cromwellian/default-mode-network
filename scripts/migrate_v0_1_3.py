"""One-shot migration from v0.1.0 / v0.1.1 / v0.1.2 SQLite profiles to v0.1.3.

What it does:

  1. Drops every existing `browser:*` interest row (manual / synthetic / readwise / etc. rows
     are left untouched).
  2. Re-imports browser history from EVERY detected browser, with no row cap, using the
     v0.1.3 importer (visit-count weighting, URL-param stripping, title-suffix stripping).
  3. Re-clusters the full profile (browser + manual + synthetic + …) with KMeans
     `sample_weight = log1p(visit_count)`, so heavy-visit-count pages pull centroids harder.
  4. Re-labels each cluster via `dmn.labeling.synthesize_label` (LLM if configured, else the
     deterministic word-frequency stub).
  5. Prints a per-browser summary, source breakdown, cluster sizes, and label samples.

Usage:

    uv run scripts/migrate_v0_1_3.py [--dry-run] [--browser-limit N]

Idempotent: running it again rebuilds from scratch, so re-running after a few weeks of
fresh browsing is the supported way to refresh the profile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/migrate_v0_1_3.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import embeddings as emb
from dmn import labeling, store, taste
from dmn.importers import browser as browser_imp
from dmn.llm import get_llm

console = Console()


def main(dry_run: bool = False, browser_limit: int | None = None) -> None:
    """Run the v0.1.3 migration in-place against `data/dmn.sqlite`."""
    conn = store.connect()  # _migrate() runs automatically on connect
    pre = conn.execute(
        "SELECT source, COUNT(*) FROM interests GROUP BY source"
    ).fetchall()
    pre_total = sum(r[1] for r in pre)
    console.print(
        f"Profile contains [bold]{pre_total}[/] interest rows before migration."
    )
    for src, n in sorted(pre, key=lambda r: -r[1]):
        console.print(f"  {n:>5}  {src}")

    # 1) Drop only browser:* rows.
    if not dry_run:
        cur = conn.execute("DELETE FROM interests WHERE source LIKE 'browser:%'")
        conn.commit()
        console.print(
            f"\n[dim]Deleted {cur.rowcount} existing browser:* interest rows.[/]"
        )

    # 2) Detect + import.
    detected = browser_imp.detected_browsers()
    if not detected:
        console.print(
            "[yellow]No browsers detected on disk. "
            "Migration finished (nothing to import).[/]"
        )
        return
    console.print(
        f"\nDetected browsers: [bold]{', '.join(detected)}[/]"
        + (f"  (limit={browser_limit})" if browser_limit else "  (unbounded)")
    )
    rows, stats = browser_imp.import_history_with_stats(
        detected, limit=browser_limit, keep_noise=False
    )

    table = Table(title="Browser import summary", show_lines=False)
    table.add_column("browser")
    table.add_column("raw", justify="right")
    table.add_column("kept", justify="right")
    table.add_column("titleless", justify="right")
    table.add_column("generic", justify="right")
    table.add_column("root", justify="right")
    table.add_column("short", justify="right")
    table.add_column("dedup", justify="right")
    for b in detected:
        s = stats.get(b, {})
        d = s.get("drops") or {}
        table.add_row(
            b,
            str(s.get("raw", 0)),
            str(s.get("kept", 0)),
            str(d.get("titleless", 0)),
            str(d.get("generic_title", 0)),
            str(d.get("root_path", 0)),
            str(d.get("short", 0)),
            str(d.get("dedup", 0)),
        )
    console.print(table)
    console.print(f"Total kept rows from all browsers: [bold]{len(rows)}[/]")

    # 3) Embed + insert browser rows.
    if rows:
        console.print("\nEmbedding browser rows…")
        texts = [r["text"] for r in rows]
        vectors = emb.embed(texts)
        if not dry_run:
            for r, v in zip(rows, vectors):
                store.add_interest(
                    conn,
                    r["text"],
                    r["source"],
                    weight=float(r.get("weight", 1.0)),
                    embedding=v.astype(np.float32),
                )

    # 4) Re-cluster the full profile (browser + manual + synthetic + …).
    interests = store.list_interests(conn)
    if not interests:
        console.print("[red]No interests left after import. Aborting.[/]")
        return

    vectors_list: list[np.ndarray] = []
    weights_list: list[float] = []
    texts_full: list[str] = []
    for i in interests:
        if i["embedding"] is not None:
            vectors_list.append(i["embedding"])
        else:
            vectors_list.append(emb.embed([i["text"]])[0])
        weights_list.append(float(i.get("weight") or 1.0))
        texts_full.append(i["text"] or "")
    full_vectors = np.stack(vectors_list).astype(np.float32)
    full_weights = np.asarray(weights_list, dtype=np.float64)

    console.print(
        f"\nClustering [bold]{len(interests)}[/] total interests "
        "(visit-count-weighted KMeans)…"
    )
    centroids, labels = taste.cluster(full_vectors, sample_weight=full_weights)

    # 5) Re-label.
    llm = get_llm(dry_run=dry_run)
    console.print(f"Synthesizing cluster labels via [bold]{llm.name}[/] LLM…")
    cluster_themes: list[str] = []
    cluster_metas: list[dict] = []
    cluster_sizes: list[int] = []
    for ci in range(len(centroids)):
        idx = [j for j, lab in enumerate(labels) if int(lab) == ci]
        cluster_sizes.append(len(idx))
        if not idx:
            cluster_themes.append(f"cluster-{ci}")
            cluster_metas.append({})
            continue
        ranked = sorted(
            idx, key=lambda j: float(np.linalg.norm(full_vectors[j] - centroids[ci]))
        )
        seen: set[str] = set()
        chosen: list[str] = []
        for j in ranked:
            t = (texts_full[j] or "").strip()
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

    if not dry_run:
        store.replace_clusters(
            conn, centroids, cluster_themes, meta_per_cluster=cluster_metas
        )
        # Update n_members so cluster sizes are visible to inspect_clusters.py.
        for ci, n in enumerate(cluster_sizes):
            conn.execute(
                "UPDATE clusters SET n_members = ? WHERE id = ?", (int(n), ci)
            )
        conn.commit()

    # Final pretty summary.
    console.print("\n[bold]Cluster summary (sorted by size):[/]")
    order = sorted(range(len(centroids)), key=lambda i: -cluster_sizes[i])
    for ci in order:
        theme = cluster_themes[ci]
        n = cluster_sizes[ci]
        meta = cluster_metas[ci] or {}
        console.print(f"  [bold cyan]{ci}[/] [dim]({n} members)[/]  {theme}")
        if meta.get("subtopics"):
            console.print(f"     subtopics: {meta['subtopics']}")

    # Source breakdown.
    rows = conn.execute(
        "SELECT source, COUNT(*), SUM(weight), AVG(weight) FROM interests GROUP BY source"
    ).fetchall()
    console.print("\n[bold]Interests by source (post-migration):[/]")
    for src, n, sw, aw in sorted(rows, key=lambda r: -r[1]):
        console.print(
            f"  {n:>5}  {src:<20}  total_weight={float(sw or 0):.1f}  "
            f"avg_weight={float(aw or 0):.2f}"
        )

    if dry_run:
        console.print("\n[yellow]--dry-run set: no changes persisted.[/]")
    else:
        console.print("\n[green]Migration v0.1.3 complete.[/]")
        console.print(
            "Run [bold]uv run scripts/inspect_clusters.py[/] to see the top members "
            "of each cluster, or [bold]uv run prepare.py --relabel-only[/] once your "
            "LLM key is set to swap the stub-frequency themes for LLM-synthesized ones."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the SQLite or calling a real LLM.",
    )
    ap.add_argument(
        "--browser-limit",
        type=int,
        default=None,
        help="Cap browser-history rows per browser (default: unbounded).",
    )
    args = ap.parse_args()
    main(dry_run=args.dry_run, browser_limit=args.browser_limit)
