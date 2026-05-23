"""DMN one-time setup: interview the user, run importers, cluster, and synthesize labels."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import __version__
from dmn import embeddings as emb
from dmn import labeling, store, taste
from dmn.importers import browser as browser_imp
from dmn.importers import drive as drive_imp
from dmn.importers import gmail as gmail_imp
from dmn.importers import manual
from dmn.importers import readwise as rw_imp
from dmn.importers import twitter as tw_imp
from dmn.importers import youtube as yt_imp
from dmn.llm import get_llm

console = Console()

BANNER = r"""
   ___  __  __  _  _
  |   \|  \/  || \| |   default-mode-network
  | |) | |\/| || .` |   the wandering mind, v{ver}
  |___/|_|  |_||_|\_|
""".strip("\n")

# How many nearest-centroid members we surface to the LLM when synthesizing a label.
# 12 is a sweet spot: enough to disambiguate, few enough that even a small local LLM
# can hold them all in context without truncation.
LABEL_SAMPLE_SIZE = 12


def _banner() -> None:
    """Print the project banner."""
    console.print(f"[bold cyan]{BANNER.format(ver=__version__)}[/]")


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
        help="Disable the v0.1.4 content-vs-service classifier "
        "(login / dashboards / checkout / internal-corp tools survive).",
    ),
    relabel_only: bool = typer.Option(
        False,
        "--relabel-only",
        help="Skip importing + clustering; just re-synthesize cluster labels against the "
        "existing profile. Fast iteration when label quality looks off.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="No API calls; install a small synthetic profile and use the stub LLM for labels.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialize data/dmn.sqlite with a taste profile (interview + importers + clustering)."""
    _banner()

    if relabel_only:
        _relabel_only(dry_run=dry_run, verbose=verbose)
        return

    interests: list[dict] = []
    sources = [s.strip() for s in import_sources.split(",") if s.strip()]

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

    console.print("Embedding…")
    texts = [i["text"] for i in interests]
    vectors = emb.embed(texts)
    weights = np.asarray(
        [float(i.get("weight", 1.0)) for i in interests], dtype=np.float64
    )

    conn = store.connect()
    for item, v in zip(interests, vectors):
        store.add_interest(
            conn,
            item["text"],
            item["source"],
            weight=float(item.get("weight", 1.0)),
            embedding=v.astype(np.float32),
        )

    console.print("Clustering taste profile (visit-count-weighted)…")
    centroids, labels = taste.cluster(vectors, sample_weight=weights)

    llm = get_llm(dry_run=dry_run)
    console.print(
        f"Synthesizing cluster labels via [bold]{llm.name}[/] LLM…"
    )
    cluster_themes, cluster_metas = _synthesize_all_labels(
        centroids, vectors, texts, labels, llm
    )
    store.replace_clusters(
        conn, centroids, cluster_themes, meta_per_cluster=cluster_metas
    )

    _print_cluster_summary(cluster_themes, cluster_metas, vectors, texts, labels, centroids)

    console.print(
        "\n[bold green]Profile ready.[/] Next: `uv run explore.py --dry-run --iterations 2`"
    )


def _relabel_only(dry_run: bool, verbose: bool) -> None:
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

    centroids = np.stack(
        [
            np.asarray(c["centroid"], dtype=np.float32)
            for c in clusters
            if c["centroid"] is not None
        ]
    )

    labels = _assign_labels(vectors, centroids)

    llm = get_llm(dry_run=dry_run)
    console.print(
        f"Re-synthesizing labels for [bold]{len(centroids)}[/] cluster(s) "
        f"via [bold]{llm.name}[/] LLM…"
    )
    cluster_themes, cluster_metas = _synthesize_all_labels(
        centroids, vectors, texts, labels, llm
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
    ci: int, vectors: np.ndarray, texts: list[str], labels: np.ndarray, centroid: np.ndarray
) -> list[str]:
    """Return up to LABEL_SAMPLE_SIZE nearest-centroid member texts for a cluster (deduped)."""
    idx = [j for j, lab in enumerate(labels) if int(lab) == ci]
    if not idx:
        return []
    ranked = sorted(idx, key=lambda j: float(np.linalg.norm(vectors[j] - centroid)))
    seen: set[str] = set()
    chosen: list[str] = []
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
) -> tuple[list[str], list[dict]]:
    """Run `labeling.synthesize_label` on every cluster's nearest-centroid members."""
    themes: list[str] = []
    metas: list[dict] = []
    for ci in range(len(centroids)):
        members = _members_for_cluster(ci, vectors, texts, labels, centroids[ci])
        if not members:
            themes.append(f"cluster-{ci}")
            metas.append({})
            continue
        cl = labeling.synthesize_label(members, llm)
        themes.append(cl.theme or f"cluster-{ci}")
        metas.append(cl.to_meta())
    return themes, metas


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
    return [{"text": s, "source": "synthetic"} for s in seeds]


if __name__ == "__main__":
    typer.run(main)
