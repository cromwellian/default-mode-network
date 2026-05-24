"""One-shot migration to v0.4: HDBSCAN clustering, recency weights, pinned manual tastes.

Idempotent upgrade on existing `data/dmn.sqlite`:
  1. Backfill `last_seen` from timestamp (manual/interview → now).
  2. Recompute composite recency × visit_count weights.
  3. Re-cluster with HDBSCAN (medoids, noise bucket).
  4. Pin manual/interview interests from noise or mega-clusters.
  5. Re-label via LLM if API key available.
  6. Print before/after cluster distribution.

Usage:

    uv run scripts/migrate_v0_4.py [--dry-run] [--cluster-method hdbscan]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

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
from dmn import store, taste
from dmn.llm import get_llm
from prepare import (
    _synthesize_all_labels,
    composite_weights_for_interests,
    persist_clusters,
    run_clustering_pipeline,
)

console = Console()


def _cluster_distribution(clusters: list[dict], total: int) -> list[tuple[int, int, float, str]]:
    """Return (id, n_members, pct, label) sorted by size descending."""
    rows = []
    for c in clusters:
        n = int(c.get("n_members") or 0)
        pct = 100.0 * n / max(total, 1)
        rows.append((c["id"], n, pct, c.get("label") or ""))
    return sorted(rows, key=lambda r: -r[1])


def _print_distribution(title: str, rows: list[tuple[int, int, float, str]], total: int) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("id", justify="right")
    table.add_column("n", justify="right")
    table.add_column("%", justify="right")
    table.add_column("label")
    for cid, n, pct, label in rows[:15]:
        table.add_row(str(cid), str(n), f"{pct:.1f}", label[:60])
    if len(rows) > 15:
        table.add_row("…", f"+{len(rows)-15}", "", "")
    console.print(table)
    console.print(f"  Total interests: {total}, clusters: {len(rows)}")


def main(
    dry_run: bool = False,
    cluster_method: str = "hdbscan",
    recency_half_life: float = 90.0,
    min_cluster_size: int | None = None,
) -> None:
    """Run the v0.4 migration in-place against `data/dmn.sqlite`."""
    conn = store.connect()
    interests = store.list_interests(conn)
    if not interests:
        console.print("[red]No interests in profile. Nothing to migrate.[/]")
        return

    pre_clusters = store.list_clusters(conn)
    pre_total = len(interests)
    pre_dist = _cluster_distribution(pre_clusters, pre_total)
    console.print(f"\n[bold]Before migration[/] ({len(pre_clusters)} clusters, {pre_total} interests)")
    _print_distribution("Pre-migration clusters", pre_dist, pre_total)
    if pre_dist:
        top_pct = pre_dist[0][2]
        console.print(f"  Largest cluster: {top_pct:.1f}% of profile")

    now = time.time()
    if not dry_run:
        for i in interests:
            ls = i.get("last_seen")
            src = i.get("source") or ""
            if ls is None:
                if src.startswith("manual") or src == "interview":
                    ls = now
                else:
                    ls = i.get("timestamp") or now
                conn.execute(
                    "UPDATE interests SET last_seen = ? WHERE id = ?",
                    (float(ls), i["id"]),
                )
        conn.commit()
        interests = store.list_interests(conn)

    vectors_list: list[np.ndarray] = []
    texts: list[str] = []
    interest_ids: list[int] = []
    interest_dicts: list[dict] = []

    for i in interests:
        if i["embedding"] is not None:
            vectors_list.append(i["embedding"])
        else:
            vectors_list.append(emb.embed([i["text"]])[0])
        texts.append(i["text"] or "")
        interest_ids.append(i["id"])
        visit_count = max(1.0, math.expm1(float(i.get("weight") or 1.0)))
        interest_dicts.append(
            {
                "text": i["text"],
                "source": i.get("source") or "",
                "weight": i.get("weight") or 1.0,
                "visit_count": visit_count,
                "last_seen": i.get("last_seen") or i.get("timestamp") or now,
                "timestamp": i.get("timestamp") or now,
            }
        )

    vectors = np.stack(vectors_list).astype(np.float32)

    console.print(
        f"\nRe-clustering [bold]{len(interests)}[/] interests "
        f"({cluster_method}, recency half-life={recency_half_life}d)…"
    )
    result, weights, synth_labels, is_noise_flags = run_clustering_pipeline(
        vectors,
        interest_dicts,
        interest_ids,
        cluster_method=cluster_method,
        recency_half_life=recency_half_life,
        min_cluster_size=min_cluster_size,
    )

    n_real = len(result.centroids) - (1 if result.n_noise > 0 else 0)
    console.print(f"Clusters (excl. noise): [bold]{n_real}[/]")
    console.print(
        f"Noise: [bold]{result.n_noise}[/] ({100.0 * result.n_noise / max(len(vectors), 1):.1f}%)"
    )
    if result.silhouette is not None:
        console.print(f"Silhouette: [bold]{result.silhouette:.3f}[/]")

    llm = get_llm(dry_run=dry_run)
    console.print(f"Synthesizing labels via [bold]{llm.name}[/]…")
    cluster_themes, cluster_metas = _synthesize_all_labels(
        result.centroids,
        vectors,
        texts,
        synth_labels,
        llm,
        interest_ids,
        result.medoid_indices,
    )

    if result.n_noise > 0 and is_noise_flags and is_noise_flags[-1]:
        if cluster_themes[-1].startswith("cluster-"):
            cluster_themes[-1] = "ambient / unclustered"
        if cluster_metas:
            cluster_metas[-1] = cluster_metas[-1] or {}
            cluster_metas[-1]["theme"] = "ambient / unclustered"

    if not dry_run:
        persist_clusters(
            conn,
            result,
            vectors,
            texts,
            interest_ids,
            synth_labels,
            weights,
            cluster_themes,
            cluster_metas,
            is_noise_flags,
        )

    post_clusters = store.list_clusters(conn) if not dry_run else [
        {
            "id": ci,
            "n_members": int(np.sum(synth_labels == ci)),
            "label": cluster_themes[ci] if ci < len(cluster_themes) else "",
            "is_noise": is_noise_flags[ci] if ci < len(is_noise_flags) else False,
        }
        for ci in range(len(result.centroids))
    ]
    post_dist = _cluster_distribution(post_clusters, pre_total)
    console.print(f"\n[bold]After migration[/]")
    _print_distribution("Post-migration clusters", post_dist, pre_total)

    if pre_dist and post_dist:
        pre_top = pre_dist[0][2]
        post_top = post_dist[0][2] if not post_dist[0][3].startswith("ambient") else (
            post_dist[1][2] if len(post_dist) > 1 else post_dist[0][2]
        )
        console.print(
            f"\nLargest cluster: {pre_top:.1f}% → {post_top:.1f}% "
            f"({'fixed' if post_top < 50 else 'still concentrated'})"
        )

    manual_rows = [
        (i["id"], i["text"][:60], i.get("source"))
        for i in interests
        if (i.get("source") or "").startswith("manual") or i.get("source") == "interview"
    ]
    if manual_rows:
        console.print("\n[bold]Manual / interview interests:[/]")
        id_to_cluster = {}
        for ci in range(len(result.centroids)):
            for j in range(len(synth_labels)):
                if int(synth_labels[j]) == ci:
                    id_to_cluster[interest_ids[j]] = (
                        cluster_themes[ci] if ci < len(cluster_themes) else f"cluster-{ci}"
                    )
        for iid, txt, src in manual_rows[:20]:
            clabel = id_to_cluster.get(iid, "?")
            console.print(f"  [{src}] → {clabel}: {txt}")

    if dry_run:
        console.print("\n[yellow]--dry-run: no changes persisted.[/]")
    else:
        console.print("\n[green]Migration v0.4 complete.[/]")
        console.print("If labels look stubby: [bold]uv run prepare.py --relabel-only[/]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--cluster-method",
        default="hdbscan",
        choices=["hdbscan", "kmeans", "gmm"],
    )
    ap.add_argument("--recency-half-life", type=float, default=90.0)
    ap.add_argument("--min-cluster-size", type=int, default=None)
    args = ap.parse_args()
    main(
        dry_run=args.dry_run,
        cluster_method=args.cluster_method,
        recency_half_life=args.recency_half_life,
        min_cluster_size=args.min_cluster_size,
    )
