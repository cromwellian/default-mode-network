"""DMN one-time setup: interview the user, run importers, cluster, and synthesize labels."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import __version__
from dmn import embeddings as emb
from dmn import labeling, portability, store, taste
from dmn.importers import browser as browser_imp
from dmn.importers import drive as drive_imp
from dmn.importers import gmail as gmail_imp
from dmn.importers import manual
from dmn.importers import readwise as rw_imp
from dmn.importers import twitter as tw_imp
from dmn.importers import youtube as yt_imp
from dmn.llm import get_llm

console = Console()


def _warn_if_stub_labels(llm, dry_run: bool, local_labels: bool) -> None:
    """Tell the user when cluster labels will be word-frequency fallbacks, not thematic."""
    if llm.name != "stub" or dry_run or local_labels:
        return
    console.print(
        "[yellow]No LLM available — cluster labels will be basic word-frequency themes. "
        "Put ANTHROPIC_API_KEY in .env and run `uv run prepare.py --relabel-only` "
        "to upgrade them later.[/]"
    )

BANNER = r"""
   ___  __  __  _  _
  |   \|  \/  || \| |   default-mode-network
  | |) | |\/| || .` |   the wandering mind, v{ver}
  |___/|_|  |_||_|\_|
""".strip("\n")

# How many nearest-centroid members we surface to the LLM when synthesizing a label.
LABEL_SAMPLE_SIZE = 12


def _banner() -> None:
    """Print the project banner."""
    console.print(f"[bold cyan]{BANNER.format(ver=__version__)}[/]")


def composite_weights_for_interests(
    interests: list[dict],
    half_life_days: float = 90.0,
) -> np.ndarray:
    """Compute composite recency × visit_count weights for clustering."""
    now = time.time()
    weights: list[float] = []
    for item in interests:
        visit_count = item.get("visit_count")
        if visit_count is None:
            visit_count = max(1.0, math.expm1(float(item.get("weight", 1.0))))
        ts = item.get("last_seen") or item.get("timestamp") or now
        days_since = max(0.0, (now - float(ts)) / 86400.0)
        w = taste.composite_weight(
            float(visit_count),
            days_since,
            item.get("source") or "",
            half_life_days=half_life_days,
        )
        weights.append(w)
    return np.asarray(weights, dtype=np.float64)


def run_clustering_pipeline(
    vectors: np.ndarray,
    interests: list[dict],
    interest_ids: list[int],
    *,
    cluster_method: str = "hdbscan",
    recency_half_life: float = 90.0,
    min_cluster_size: Optional[int] = None,
) -> tuple[taste.ClusterResult, np.ndarray, np.ndarray, list[bool]]:
    """Cluster, pin manual tastes, add noise bucket; return result + cluster-index labels."""
    weights = composite_weights_for_interests(interests, half_life_days=recency_half_life)
    sources = [i.get("source") or "" for i in interests]

    result = taste.cluster(
        vectors,
        sample_weight=weights,
        method=cluster_method,
        recency_weights=weights,
        min_cluster_size=min_cluster_size,
    )
    result, weights = taste.pin_manual_interests(
        result, vectors, sources, weights
    )
    result = taste.add_noise_cluster(result, vectors, weights)

    unique_labels = sorted(set(int(l) for l in result.labels))
    label_to_ci = {lab: ci for ci, lab in enumerate(unique_labels)}
    synth_labels = np.array([label_to_ci[int(l)] for l in result.labels], dtype=int)

    is_noise_flags = [False] * len(unique_labels)
    if result.n_noise > 0:
        is_noise_flags[-1] = True

    return result, weights, synth_labels, is_noise_flags


def persist_clusters(
    conn,
    result: taste.ClusterResult,
    vectors: np.ndarray,
    texts: list[str],
    interest_ids: list[int],
    synth_labels: np.ndarray,
    weights: np.ndarray,
    cluster_themes: list[str],
    cluster_metas: list[dict],
    is_noise_flags: list[bool],
    delete_interest_ids: Optional[list[int]] = None,
) -> None:
    """Write clusters to SQLite with correct n_members and medoid ids.

    `delete_interest_ids` (profile replacement) is applied in the same transaction
    as the cluster swap, so an interrupt can't leave clusters pointing at deleted
    interest rows.
    """
    n = len(vectors)
    unique_labels = sorted(set(int(l) for l in result.labels))
    n_clusters = len(result.centroids)
    n_members: list[int] = []
    medoid_ids: list[int | None] = []
    metas = [dict(m or {}) for m in cluster_metas]

    for ci in range(n_clusters):
        lab = unique_labels[ci] if ci < len(unique_labels) else ci
        members = [j for j in range(n) if int(result.labels[j]) == lab]
        n_members.append(len(members))
        midx = (
            result.medoid_indices[ci]
            if ci < len(result.medoid_indices)
            else (members[0] if members else 0)
        )
        medoid_ids.append(interest_ids[midx] if midx < len(interest_ids) else None)
        mass = sum(float(weights[j]) for j in members)
        if ci < len(metas):
            metas[ci]["cluster_mass"] = mass
            if result.silhouette is not None:
                metas[ci]["silhouette"] = result.silhouette
        else:
            meta = {"cluster_mass": mass}
            if result.silhouette is not None:
                meta["silhouette"] = result.silhouette
            metas.append(meta)

    if delete_interest_ids:
        # No commit here: replace_clusters commits, closing one atomic swap.
        conn.executemany(
            "DELETE FROM interests WHERE id = ?",
            [(int(i),) for i in delete_interest_ids],
        )
    store.replace_clusters(
        conn,
        result.centroids,
        cluster_themes,
        meta_per_cluster=metas,
        n_members=n_members,
        medoid_interest_ids=medoid_ids,
        cluster_method=result.method,
        is_noise_flags=is_noise_flags,
    )
    portability.stamp_profile_embedding(conn)


def main(
    interactive: bool = typer.Option(
        False, "--interactive", help="Run the manual taste interview."
    ),
    import_sources: str = typer.Option(
        "",
        "--import",
        help="Comma-separated importers: browser,youtube,gmail,drive,twitter,readwise",
    ),
    takeout_dir: Optional[Path] = typer.Option(
        None, "--takeout-dir", help="Path to a Google Takeout directory."
    ),
    twitter_dir: Optional[Path] = typer.Option(
        None, "--twitter-dir", help="Path to a Twitter/X data export directory."
    ),
    readwise_csv: Optional[Path] = typer.Option(
        None, "--readwise-csv", help="Path to a Readwise / Pocket / GoodReads CSV."
    ),
    browsers: str = typer.Option(
        "",
        "--browsers",
        help="Comma-separated browsers (chrome,arc,brave,edge,firefox,safari). "
        "Empty = auto-detect every browser whose history db exists on disk.",
    ),
    browser_limit: Optional[int] = typer.Option(
        None,
        "--browser-limit",
        help="Cap browser-history rows per browser. Default = unbounded.",
    ),
    keep_noise: bool = typer.Option(
        False,
        "--keep-noise",
        help="Disable the browser-history filter (homepage / login / generic-domain dedup).",
    ),
    keep_services: bool = typer.Option(
        False,
        "--keep-services",
        help="Disable the v0.1.4 content-vs-service classifier.",
    ),
    relabel_only: bool = typer.Option(
        False,
        "--relabel-only",
        help="Skip importing + clustering; just re-synthesize cluster labels.",
    ),
    cluster_method: str = typer.Option(
        "hdbscan",
        "--cluster-method",
        help="Clustering method: hdbscan, kmeans, or gmm.",
    ),
    recency_half_life: float = typer.Option(
        90.0,
        "--recency-half-life",
        help="Half-life in days for recency decay in composite weights.",
    ),
    min_cluster_size: Optional[int] = typer.Option(
        None,
        "--min-cluster-size",
        help="Override HDBSCAN min_cluster_size (default: max(15, n//80)).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="No API calls; install a small synthetic profile and use the stub LLM for labels.",
    ),
    local_labels: bool = typer.Option(
        False,
        "--local-labels",
        help="Do not send sampled interest text to a remote LLM for cluster labels.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an existing profile without prompting (the journal is kept).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialize data/dmn.sqlite with a taste profile (interview + importers + clustering)."""
    _banner()

    if relabel_only:
        _relabel_only(dry_run=dry_run, local_labels=local_labels, verbose=verbose)
        return

    interests: list[dict] = []
    sources = [s.strip() for s in import_sources.split(",") if s.strip()]
    known_sources = {"browser", "youtube", "gmail", "drive", "twitter", "readwise"}
    unknown_sources = [s for s in sources if s not in known_sources]
    if unknown_sources:
        console.print(
            f"[red]Unknown --import source(s): {', '.join(unknown_sources)}. "
            f"Valid: {', '.join(sorted(known_sources))}.[/]"
        )
        raise typer.Exit(code=1)

    if dry_run and not interactive and not sources:
        console.print(
            "[yellow]--dry-run with no inputs: seeding a small synthetic profile.[/]"
        )
        interests.extend(_synthetic_profile())

    if interactive:
        console.print("[bold]Interview mode.[/]")
        interests.extend(manual.interview())

    if "browser" in sources:
        browser_list = (
            [b.strip() for b in browsers.split(",") if b.strip()]
            if browsers
            else browser_imp.detected_browsers()
        )
        if not browser_list:
            console.print(
                "[yellow]No browsers detected on disk; skipping browser import.[/]"
            )
        else:
            console.print(
                f"Importing browser history from: [bold]{', '.join(browser_list)}[/]"
                + (f" (limit={browser_limit})" if browser_limit else "")
            )
            interests.extend(
                browser_imp.import_history(
                    browser_list,
                    limit=browser_limit,
                    keep_noise=keep_noise,
                    keep_services=keep_services,
                )
            )
    if "youtube" in sources:
        if not takeout_dir:
            console.print(
                "[yellow]--import youtube needs --takeout-dir; skipping.[/]"
            )
        else:
            console.print("Importing YouTube watch history…")
            interests.extend(yt_imp.import_watch_history(takeout_dir))
    if "gmail" in sources:
        if not takeout_dir:
            console.print("[yellow]--import gmail needs --takeout-dir; skipping.[/]")
        else:
            console.print("Importing Gmail Sent…")
            interests.extend(gmail_imp.import_sent(takeout_dir))
    if "drive" in sources:
        if not takeout_dir:
            console.print("[yellow]--import drive needs --takeout-dir; skipping.[/]")
        else:
            console.print("Importing Drive titles…")
            interests.extend(drive_imp.import_drive_titles(takeout_dir))
    if "twitter" in sources:
        if not twitter_dir:
            console.print(
                "[yellow]--import twitter needs --twitter-dir; skipping.[/]"
            )
        else:
            console.print("Importing Twitter likes…")
            interests.extend(tw_imp.import_likes(twitter_dir))
    if "readwise" in sources:
        if not readwise_csv:
            console.print(
                "[yellow]--import readwise needs --readwise-csv; skipping.[/]"
            )
        else:
            console.print("Importing Readwise CSV…")
            interests.extend(rw_imp.import_csv(readwise_csv))

    interests = _dedupe(interests)

    if not interests:
        console.print(
            "[red]No interests collected. Re-run with --interactive or --import …[/]"
        )
        raise typer.Exit(code=1)

    console.print(f"Collected [bold]{len(interests)}[/] interest items.")
    if verbose:
        for item in interests[:20]:
            console.print(f"  · [{item['source']}] {item['text'][:120]}")

    conn = store.connect()
    existing = store.list_interests(conn)
    old_interest_ids: list[int] = []
    if existing:
        console.print(
            f"[yellow]This replaces your existing profile ({len(existing)} interests) "
            f"with the {len(interests)} item(s) just collected. Your journal is kept.[/]"
        )
        if not replace:
            if not sys.stdin.isatty():
                console.print(
                    "[red]Refusing to replace an existing profile non-interactively. "
                    "Pass --replace to rebuild.[/]"
                )
                conn.close()
                raise typer.Exit(1)
            if not typer.confirm("Replace it?", default=False):
                console.print("Kept the existing profile — nothing was changed.")
                conn.close()
                raise typer.Exit(0)
        old_interest_ids = [int(r["id"]) for r in existing]

    texts = [i["text"] for i in interests]
    emb.embed(texts[:1])  # first call prints any fallback notice cleanly, pre-spinner
    with console.status(f"Embedding {len(texts)} interest(s)…"):
        vectors = emb.embed(texts)

    interest_ids: list[int] = []
    for item, v in zip(interests, vectors):
        iid = store.add_interest(
            conn,
            item["text"],
            item["source"],
            weight=float(item.get("weight", 1.0)),
            embedding=v.astype(np.float32),
            last_seen=item.get("last_seen"),
        )
        interest_ids.append(iid)

    with console.status(f"Clustering taste profile ({cluster_method}, recency-weighted)…"):
        result, weights, synth_labels, is_noise_flags = run_clustering_pipeline(
            vectors,
            interests,
            interest_ids,
            cluster_method=cluster_method,
            recency_half_life=recency_half_life,
            min_cluster_size=min_cluster_size,
        )
    if result.method != cluster_method:
        console.print(
            f"[yellow]{cluster_method} found no dense clusters here "
            f"(common for small profiles) — used {result.method} instead.[/]"
        )

    if result.silhouette is not None:
        console.print(f"Silhouette score (non-noise): [bold]{result.silhouette:.3f}[/]")
    console.print(
        f"Noise points: [bold]{result.n_noise}[/] "
        f"({100.0 * result.n_noise / max(len(vectors), 1):.1f}%)"
    )

    llm = get_llm(dry_run=True) if local_labels else get_llm(dry_run=dry_run)
    _warn_if_stub_labels(llm, dry_run=dry_run, local_labels=local_labels)
    console.print(
        f"Synthesizing cluster labels via [bold]{llm.name}[/] LLM…"
    )
    cluster_themes, cluster_metas = _synthesize_all_labels(
        result.centroids, vectors, texts, synth_labels, llm, interest_ids, result.medoid_indices
    )

    if result.n_noise > 0 and is_noise_flags and is_noise_flags[-1]:
        if cluster_themes[-1].startswith("cluster-"):
            cluster_themes[-1] = "ambient / unclustered"
        if cluster_metas:
            cluster_metas[-1] = cluster_metas[-1] or {}
            cluster_metas[-1]["theme"] = "ambient / unclustered"

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
        delete_interest_ids=old_interest_ids,
    )
    store.clear_profile_stale(conn)

    _print_cluster_table(
        result, vectors, texts, synth_labels, cluster_themes, result.medoid_indices
    )

    console.print(
        "\n[bold green]Profile ready.[/] Next: `uv run explore.py --dry-run --iterations 2`"
    )


def _relabel_only(dry_run: bool, local_labels: bool, verbose: bool) -> None:
    """Re-synthesize cluster labels against the existing profile, skipping import + reclustering."""
    conn = store.connect()
    interests = store.list_interests(conn)
    clusters = store.list_clusters(conn)
    if not interests or not clusters:
        console.print(
            "[red]Nothing to relabel: missing interests or clusters. "
            "Run `uv run prepare.py` first to build a profile.[/]"
        )
        raise typer.Exit(1)

    vectors = np.stack(
        [
            i["embedding"]
            if i["embedding"] is not None
            else emb.embed([i["text"]])[0]
            for i in interests
        ]
    ).astype(np.float32)
    texts = [i["text"] for i in interests]
    interest_ids = [i["id"] for i in interests]

    centroids = np.stack(
        [
            np.asarray(c["centroid"], dtype=np.float32)
            for c in clusters
            if c["centroid"] is not None
        ]
    )
    medoid_indices = [
        next(
            (j for j, i in enumerate(interests) if i["id"] == c.get("medoid_interest_id")),
            0,
        )
        for c in clusters
        if c["centroid"] is not None
    ]

    labels = _assign_labels(vectors, centroids)

    llm = get_llm(dry_run=True) if local_labels else get_llm(dry_run=dry_run)
    _warn_if_stub_labels(llm, dry_run=dry_run, local_labels=local_labels)
    console.print(
        f"Re-synthesizing labels for [bold]{len(centroids)}[/] cluster(s) "
        f"via [bold]{llm.name}[/] LLM…"
    )
    cluster_themes, cluster_metas = _synthesize_all_labels(
        centroids, vectors, texts, labels, llm, interest_ids, medoid_indices
    )
    for ci, (theme, meta) in enumerate(zip(cluster_themes, cluster_metas)):
        if ci < len(clusters):
            store.update_cluster_label(conn, clusters[ci]["id"], theme)
            store.update_cluster_meta(conn, clusters[ci]["id"], meta)

    _print_cluster_summary(cluster_themes, cluster_metas, vectors, texts, labels, centroids)
    console.print("\n[bold green]Re-labeled.[/]")


def _dedupe(items: list[dict]) -> list[dict]:
    """Remove exact-text duplicates while preserving order."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        key = (it.get("text") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _assign_labels(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each vector to the nearest centroid (Euclidean). Returns int labels."""
    if len(centroids) == 0 or len(vectors) == 0:
        return np.array([], dtype=int)
    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=-1)
    return dists.argmin(axis=1)


def _members_for_cluster(
    ci: int,
    vectors: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    centroid: np.ndarray,
    medoid_idx: Optional[int] = None,
) -> list[str]:
    """Return up to LABEL_SAMPLE_SIZE member texts for a cluster (medoid first)."""
    idx = [j for j, lab in enumerate(labels) if int(lab) == ci]
    if not idx:
        return []
    seen: set[str] = set()
    chosen: list[str] = []
    if medoid_idx is not None and medoid_idx in idx:
        t = (texts[medoid_idx] or "").strip()
        if t:
            chosen.append(t[:200])
            seen.add(t.lower())
    ranked = sorted(idx, key=lambda j: float(np.linalg.norm(vectors[j] - centroid)))
    for j in ranked:
        t = (texts[j] or "").strip()
        key = t.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append(t[:200])
        if len(chosen) >= LABEL_SAMPLE_SIZE:
            break
    return chosen


def _synthesize_all_labels(
    centroids: np.ndarray,
    vectors: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    llm,
    interest_ids: Optional[list[int]] = None,
    medoid_indices: Optional[list[int]] = None,
) -> tuple[list[str], list[dict]]:
    """Run `labeling.synthesize_label` on every cluster's nearest-centroid members."""
    themes: list[str] = []
    metas: list[dict] = []
    for ci in range(len(centroids)):
        midx = medoid_indices[ci] if medoid_indices and ci < len(medoid_indices) else None
        members = _members_for_cluster(ci, vectors, texts, labels, centroids[ci], midx)
        if not members:
            themes.append(f"cluster-{ci}")
            metas.append({})
            continue
        cl = labeling.synthesize_label(members, llm)
        themes.append(cl.theme or f"cluster-{ci}")
        metas.append(cl.to_meta())
    return themes, metas


def _print_cluster_table(
    result: taste.ClusterResult,
    vectors: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    themes: list[str],
    medoid_indices: list[int],
) -> None:
    """Print cluster distribution table with medoid text and member counts."""
    n = len(vectors)
    table = Table(title="Cluster distribution", show_lines=False)
    table.add_column("id", justify="right")
    table.add_column("n", justify="right")
    table.add_column("%", justify="right")
    table.add_column("medoid (truncated)")
    table.add_column("label")

    order = sorted(range(len(result.centroids)), key=lambda ci: -int(np.sum(labels == ci)))
    for ci in order:
        count = int(np.sum(labels == ci))
        pct = 100.0 * count / max(n, 1)
        midx = medoid_indices[ci] if ci < len(medoid_indices) else 0
        medoid_text = (texts[midx] if midx < len(texts) else "")[:80]
        theme = themes[ci] if ci < len(themes) else f"cluster-{ci}"
        table.add_row(str(ci), str(count), f"{pct:.1f}", medoid_text, theme)
    console.print()
    console.print(table)


def _print_cluster_summary(
    themes: list[str],
    metas: list[dict],
    vectors: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    centroids: np.ndarray,
) -> None:
    """Pretty-print each synthesized theme + a few example members for sanity checking."""
    console.print(f"\n[bold]Synthesized [cyan]{len(themes)}[/] cluster theme(s):[/]")
    for ci, (theme, meta) in enumerate(zip(themes, metas)):
        members = _members_for_cluster(ci, vectors, texts, labels, centroids[ci])
        console.print(f"  [bold cyan]{ci}.[/] {theme}")
        if meta and meta.get("subtopics"):
            console.print(f"     subtopics: {meta['subtopics']}")
        for m in members[:3]:
            console.print(f"       · {m[:90]}")


def _synthetic_profile() -> list[dict]:
    """A handful of seeded interests for --dry-run smoke tests."""
    seeds = [
        "the default mode network in human cognition",
        "Lisp macros and metaprogramming",
        "lo-fi hip hop and field recordings",
        "polyrhythms in jazz fusion",
        "history of color in renaissance painting",
        "swarm intelligence and ant colony optimization",
        "mid-century modern architecture",
        "the sociology of internet subcultures",
        "transformer interpretability and circuits",
        "cooking with miso and koji fermentation",
        "the design language of early Macintosh icons",
        "cosmic-ray detection in old photographs",
    ]
    return [{"text": s, "source": "synthetic", "visit_count": 1} for s in seeds]


if __name__ == "__main__":
    typer.run(main)
