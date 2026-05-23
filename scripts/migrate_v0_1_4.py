"""One-shot migration from v0.1.0 / v0.1.1 / v0.1.2 / v0.1.3 SQLite profiles to v0.1.4.

What it does:

  1. Drops every existing `browser:*` interest row (manual / synthetic / readwise / etc.
     are left untouched).
  2. Re-imports browser history from EVERY detected browser, with no row cap, using the
     v0.1.4 importer (visit-count weighting + URL-param stripping + title-suffix stripping
     + content-vs-service classifier + per-host service-ratio gate).
  3. Re-clusters the full profile with KMeans `sample_weight = log1p(visit_count)`.
  4. Re-labels each cluster via `dmn.labeling.synthesize_label` (LLM if configured, else
     deterministic word-frequency stub).
  5. Prints per-browser drop tables, service-host gate dropoffs, source breakdown, cluster
     summary with service:content ratios, and (optionally) calls `inspect_clusters.py`.

Usage:

    uv run scripts/migrate_v0_1_4.py [--dry-run] [--keep-services] [--browser-limit N]

`--keep-services` skips the content-vs-service classifier (preserves login flows /
dashboards / checkout / internal-corp tools as interests).
"""
from __future__ import annotations

import argparse
import sys
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
from dmn import labeling, store, taste
from dmn.importers import browser as browser_imp
from dmn.importers._classify import is_content_url
from dmn.llm import get_llm

console = Console()


def main(
    dry_run: bool = False,
    browser_limit: int | None = None,
    keep_services: bool = False,
) -> None:
    """Run the v0.1.4 migration in-place against `data/dmn.sqlite`."""
    conn = store.connect()
    pre = conn.execute(
        "SELECT source, COUNT(*) FROM interests GROUP BY source"
    ).fetchall()
    pre_total = sum(r[1] for r in pre)
    console.print(
        f"Profile contains [bold]{pre_total}[/] interest rows before migration."
    )
    for src, n in sorted(pre, key=lambda r: -r[1]):
        console.print(f"  {n:>5}  {src}")

    if not dry_run:
        cur = conn.execute("DELETE FROM interests WHERE source LIKE 'browser:%'")
        conn.commit()
        console.print(
            f"\n[dim]Deleted {cur.rowcount} existing browser:* interest rows.[/]"
        )

    detected = browser_imp.detected_browsers()
    if not detected:
        console.print(
            "[yellow]No browsers detected on disk. Migration finished (nothing to import).[/]"
        )
        return
    flag = "  --keep-services" if keep_services else "  (classifier active)"
    console.print(
        f"\nDetected browsers: [bold]{', '.join(detected)}[/]"
        + (f"  (limit={browser_limit})" if browser_limit else "  (unbounded)")
        + flag
    )
    rows, stats = browser_imp.import_history_with_stats(
        detected,
        limit=browser_limit,
        keep_noise=False,
        keep_services=keep_services,
    )

    table = Table(title="Browser import summary", show_lines=False)
    table.add_column("browser")
    table.add_column("raw", justify="right")
    table.add_column("titleless", justify="right")
    table.add_column("generic", justify="right")
    table.add_column("root", justify="right")
    table.add_column("short", justify="right")
    table.add_column("service", justify="right", style="bold")
    table.add_column("host_gate", justify="right", style="bold")
    table.add_column("dedup", justify="right")
    table.add_column("kept", justify="right", style="green")
    for b in detected:
        s = stats.get(b, {})
        d = s.get("drops") or {}
        table.add_row(
            b,
            str(s.get("raw", 0)),
            str(d.get("titleless", 0)),
            str(d.get("generic_title", 0)),
            str(d.get("root_path", 0)),
            str(d.get("short", 0)),
            str(d.get("service", 0)),
            str(d.get("service_host_gate", 0)),
            str(d.get("dedup", 0)),
            str(s.get("kept", 0)),
        )
    console.print(table)
    console.print(f"Total kept rows from all browsers (post-canonical-dedup): [bold]{len(rows)}[/]")

    # Service-host gate dropoffs.
    all_gate: list[dict] = []
    for b in detected:
        for entry in stats.get(b, {}).get("service_host_gate", []) or []:
            all_gate.append({**entry, "browser": b})
    if all_gate:
        console.print("\n[bold]Service hosts dropped wholesale by per-host gate:[/]")
        for g in sorted(all_gate, key=lambda d: -d["dropped"])[:20]:
            console.print(
                f"  {g['dropped']:>4} rows  ratio={g['ratio']:.2f}  "
                f"({g['service']}/{g['total']}) "
                f"[dim]{g['host']}  ({g['browser']})[/]"
            )
        if len(all_gate) > 20:
            console.print(f"  [dim]… and {len(all_gate) - 20} more[/]")

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

    llm = get_llm(dry_run=dry_run)
    console.print(f"Synthesizing cluster labels via [bold]{llm.name}[/] LLM…")
    cluster_themes: list[str] = []
    cluster_metas: list[dict] = []
    cluster_sizes: list[int] = []
    cluster_member_texts: list[list[str]] = []
    for ci in range(len(centroids)):
        idx = [j for j, lab in enumerate(labels) if int(lab) == ci]
        cluster_sizes.append(len(idx))
        if not idx:
            cluster_themes.append(f"cluster-{ci}")
            cluster_metas.append({})
            cluster_member_texts.append([])
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
        cluster_member_texts.append([texts_full[j] for j in idx])

    if not dry_run:
        store.replace_clusters(
            conn, centroids, cluster_themes, meta_per_cluster=cluster_metas
        )
        for ci, n in enumerate(cluster_sizes):
            conn.execute(
                "UPDATE clusters SET n_members = ? WHERE id = ?", (int(n), ci)
            )
        conn.commit()

    # Cluster summary with service:content ratio.
    console.print("\n[bold]Cluster summary (sorted by size, with service:content ratio):[/]")
    order = sorted(range(len(centroids)), key=lambda i: -cluster_sizes[i])
    for ci in order:
        theme = cluster_themes[ci]
        n = cluster_sizes[ci]
        meta = cluster_metas[ci] or {}
        ratio, n_content, n_service = _service_ratio_of(cluster_member_texts[ci])
        console.print(
            f"  [bold cyan]{ci}[/] [dim]({n} members)[/]  {theme}  "
            f"[dim]content/service = {n_content}/{n_service}  ratio={ratio:.2f}[/]"
        )
        if meta.get("subtopics"):
            console.print(f"     subtopics: {meta['subtopics']}")

    rows = conn.execute(
        "SELECT source, COUNT(*), SUM(weight), AVG(weight) FROM interests GROUP BY source"
    ).fetchall()
    console.print("\n[bold]Interests by source (post-migration):[/]")
    for src, n, sw, aw in sorted(rows, key=lambda r: -r[1]):
        console.print(
            f"  {n:>5}  {src:<22}  total_weight={float(sw or 0):.1f}  "
            f"avg_weight={float(aw or 0):.2f}"
        )

    if dry_run:
        console.print("\n[yellow]--dry-run set: no changes persisted.[/]")
    else:
        console.print("\n[green]Migration v0.1.4 complete.[/]")
        console.print(
            "Run [bold]uv run scripts/inspect_clusters.py --top 5 --width 100[/] for details."
        )


def _service_ratio_of(texts: list[str]) -> tuple[float, int, int]:
    """Classify a cluster's member texts; return (service_ratio, n_content, n_service)."""
    import re as _re

    pat = _re.compile(r"^(.*) \((https?://[^)]+)\)$")
    n_content = n_service = 0
    for t in texts:
        m = pat.match(t or "")
        title, url = (m.group(1), m.group(2)) if m else ((t or ""), "")
        if not url:
            n_content += 1
            continue
        ok, _ = is_content_url(url, title)
        if ok:
            n_content += 1
        else:
            n_service += 1
    total = n_content + n_service
    return ((n_service / total) if total else 0.0, n_content, n_service)


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
    ap.add_argument(
        "--keep-services",
        action="store_true",
        help="Skip the content-vs-service classifier (preserves login / dashboards / checkout).",
    )
    args = ap.parse_args()
    main(
        dry_run=args.dry_run,
        browser_limit=args.browser_limit,
        keep_services=args.keep_services,
    )
