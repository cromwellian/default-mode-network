"""Print the top-N nearest-centroid members for each cluster in `data/dmn.sqlite`.

This is the right tool for eyeballing whether your taste clusters are thematically coherent
(the labels can be re-synthesized whenever — but bad clustering means bad seeds forever).

Usage:

    uv run scripts/inspect_clusters.py            # 5 members per cluster
    uv run scripts/inspect_clusters.py --top 10   # 10 members per cluster
    uv run scripts/inspect_clusters.py --width 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import re

import numpy as np
from rich.console import Console

from dmn import store
from dmn.importers._classify import is_content_url

console = Console()

_TITLE_URL_RE = re.compile(r"^(.*) \((https?://[^)]+)\)$")


def _split_text(text: str) -> tuple[str, str]:
    """Pull (title, url) out of an interest's stored 'title (url)' text representation."""
    m = _TITLE_URL_RE.match(text or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return (text or "").strip(), ""


def _service_ratio(member_texts: list[str]) -> tuple[float, int, int]:
    """Classify each member as content/service via `is_content_url`; return (ratio, content, service)."""
    if not member_texts:
        return 0.0, 0, 0
    content = 0
    service = 0
    for t in member_texts:
        title, url = _split_text(t)
        if not url:
            content += 1
            continue
        ok, _reason = is_content_url(url, title)
        if ok:
            content += 1
        else:
            service += 1
    total = content + service
    return (service / total) if total else 0.0, content, service


def main(top: int, width: int) -> None:
    """For each cluster, print up to `top` interests closest to its centroid + service:content ratio."""
    conn = store.connect()
    interests = store.list_interests(conn)
    clusters = store.list_clusters(conn)
    if not clusters:
        console.print("[red]No clusters in profile. Run `uv run prepare.py` first.[/]")
        return
    if not interests:
        console.print("[red]No interests in profile.[/]")
        return

    vectors = np.stack(
        [
            i["embedding"]
            if i["embedding"] is not None
            else np.zeros_like(clusters[0]["centroid"], dtype=np.float32)
            for i in interests
        ]
    ).astype(np.float32)

    # Pre-compute, for every interest, which cluster it actually belongs to.
    valid_centroids = [c["centroid"] for c in clusters if c["centroid"] is not None]
    if not valid_centroids:
        return
    cs = np.stack(valid_centroids).astype(np.float32)
    cluster_ids = [c["id"] for c in clusters if c["centroid"] is not None]
    own_cluster: list[int] = []
    for j in range(len(vectors)):
        d = np.linalg.norm(cs - vectors[j], axis=1)
        own_cluster.append(cluster_ids[int(np.argmin(d))])

    for c in sorted(clusters, key=lambda c: -int(c.get("n_members") or 0)):
        cid = c["id"]
        label = c.get("label") or f"cluster-{cid}"
        n = c.get("n_members") or 0
        centroid = c["centroid"]
        if centroid is None:
            continue

        member_idx = [j for j in range(len(interests)) if own_cluster[j] == cid]
        member_texts = [interests[j].get("text") or "" for j in member_idx]
        ratio, n_content, n_service = _service_ratio(member_texts)

        dists = np.linalg.norm(vectors[member_idx] - centroid, axis=1) if member_idx else np.array([])
        order = list(np.argsort(dists))
        top_idx = [member_idx[k] for k in order[:top]]

        console.print(
            f"\n[bold cyan]cluster {cid}[/] [dim]({n} members)[/]  {label}  "
            f"[dim]content/service = {n_content}/{n_service}  ratio={ratio:.2f}[/]"
        )
        meta = c.get("meta") or {}
        if meta.get("subtopics"):
            console.print(f"   subtopics: {meta['subtopics']}")
        for j in top_idx:
            t = (interests[j].get("text") or "").strip()
            t_short = t if len(t) <= width else t[: width - 1] + "\u2026"
            console.print(f"   \u00b7 {t_short}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--top", type=int, default=5, help="members per cluster to print")
    ap.add_argument("--width", type=int, default=80, help="truncate each member to N chars")
    args = ap.parse_args()
    main(top=args.top, width=args.width)
