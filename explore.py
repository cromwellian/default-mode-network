"""DMN exploration loop — the agent-modifiable file. Iterate on this to find better strategies.

# rationale: v0.1.2 — seeds use cluster.meta.theme / .subtopics (synthesized by labeling.py)
#                     instead of raw labels; cross_pollinate picks the most-distant pair;
#                     journal frontmatter records seed_subtopic when applicable.
# rationale: v0.1.1 — local LLM providers (ollama/lmstudio), generative-media artifacts via
#                     --generate / --modalities, --minutes default bumped to 12.
# rationale: v0.1   — baseline mix of cold/cluster/cross/drift seeds; tools picked by LLM.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import __version__
from dmn import embeddings as emb
from dmn import generators as gens
from dmn import journal, seeds, store, taste
from dmn.llm import get_llm
from dmn.loop import (
    SAFE_TOOLS,
    SYNTHESIS_SYSTEM,
    execute_tools,
    parse_synthesis,
    plan_tools,
)
from dmn.tools import ResearchItem

console = Console()

BANNER = r"""
   ___  __  __  _  _
  |   \|  \/  || \| |   default-mode-network
  | |) | |\/| || .` |   wandering... v{ver}
  |___/|_|  |_||_|\_|
""".strip("\n")


def main(
    # 12 minutes is DMN's sweet spot for a wander: each iteration is multi-tool research
    # plus an LLM synthesis pass, so the per-step cost is much higher than autoresearch's
    # single 5-minute training run. Drop to ~5 for a quick browse, push to 30+ for a soak.
    minutes: float = typer.Option(
        12.0,
        "--minutes",
        "-m",
        help="Wall-clock budget. 10-15 min is the sweet spot.",
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Hard cap on the number of briefs (alternative to --minutes).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Stub LLM and only no-auth tools."
    ),
    seed_text: Optional[str] = typer.Option(
        None,
        "--seed",
        help="Single explicit seed (overrides generators for the first iter).",
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Also call generative-media tools (image/music/video) per brief.",
    ),
    modalities: str = typer.Option(
        "image,music,video",
        "--modalities",
        help="CSV of modalities to enable when --generate is set.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a wandering-mind research session and write briefs to journal/."""
    console.print(f"[bold magenta]{BANNER.format(ver=__version__)}[/]")
    rng = random.Random()

    llm = get_llm(dry_run=dry_run)
    console.print(f"LLM provider: [bold]{llm.name}[/]")

    enabled_modalities = {m.strip() for m in modalities.split(",") if m.strip()}
    if generate:
        console.print(
            f"Generators: [bold]on[/] (modalities: {sorted(enabled_modalities)})"
        )

    conn = store.connect()
    interests = store.list_interests(conn)
    clusters = store.list_clusters(conn)
    if not interests:
        console.print(
            "[red]No taste profile found. Run `uv run prepare.py --interactive` "
            "(or `--dry-run`) first.[/]"
        )
        raise typer.Exit(1)

    centroids: list[np.ndarray] = [
        c["centroid"] for c in clusters if c["centroid"] is not None
    ]
    recent_findings = store.list_recent_findings(conn, limit=50)
    recent_embs: list[np.ndarray] = [
        r["embedding"] for r in recent_findings if r["embedding"] is not None
    ]

    deadline = time.time() + minutes * 60 if minutes else None
    n_done = 0

    while True:
        if deadline is not None and time.time() >= deadline:
            break
        if iterations is not None and n_done >= iterations:
            break

        if seed_text and n_done == 0:
            chosen_seed = seeds.Seed(text=seed_text, source="manual")
        else:
            chosen_seed = _pick_seed(
                clusters, llm, rng, store.list_journal(conn, limit=10)
            )

        console.print(
            Panel(
                f"[bold]Seed:[/] {chosen_seed.text}\n"
                f"[dim]source: {chosen_seed.source}[/]",
                border_style="cyan",
            )
        )

        plan = plan_tools(chosen_seed.text, llm, dry_run, rng)
        console.print(f"  using tools: {plan or '(none available)'}")

        items = execute_tools(
            plan,
            chosen_seed.text,
            verbose=verbose,
            log=console.print if verbose else None,
        )
        if not items:
            console.print("[yellow]  (no results, skipping)[/]")
            n_done += 1
            continue

        item_embs = emb.embed(
            [(i.title or "") + " — " + (i.summary or "") for i in items]
        )
        for it, v in zip(items, item_embs):
            it.embedding = v

        scored: list[tuple[float, dict, ResearchItem]] = []
        for it in items:
            d = taste.dopamine(it.embedding, centroids, recent_embs, rng)
            scored.append((d["total"], d, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:5]

        bullets: list[str] = []
        for _, d, it in top_k:
            bullets.append(
                f"- [{it.source}] **{it.title}** — {it.summary[:240]}\n"
                f"  {it.url}\n"
                f"  dopamine: {d}"
            )
        gathered = "\n".join(bullets)
        prompt = (
            f"Seed question: {chosen_seed.text}\n\n"
            f"Gathered findings:\n{gathered}\n\n"
            "Write the brief now."
        )
        try:
            resp = llm.complete(
                system=SYNTHESIS_SYSTEM, user=prompt, max_tokens=800
            )
            raw_body = (resp.text or "").strip()
        except Exception as e:
            if verbose:
                console.print(f"[yellow]  LLM error: {e}[/]")
            raw_body = f"_(synthesis failed; raw findings)_\n\n{gathered}"
        body, entities, rabbit_holes = parse_synthesis(raw_body)

        brief_emb = emb.embed([chosen_seed.text + " :: " + body[:1000]])[0]
        d_brief = taste.dopamine(brief_emb, centroids, recent_embs, rng)

        artifact_meta: Optional[dict] = None
        if generate:
            label = _label_for_top_cluster(clusters, centroids, brief_emb)
            artifact_meta = _maybe_generate(
                label, chosen_seed.text, body, enabled_modalities, dry_run, verbose
            )

        path = journal.write_brief(
            seed=chosen_seed.text,
            body=body,
            dopamine=d_brief,
            seed_source=chosen_seed.source,
            seed_subtopic=getattr(chosen_seed, "subtopic", None),
            tools=plan,
            artifact=artifact_meta,
        )
        store.add_journal(
            conn,
            chosen_seed.text,
            chosen_seed.source,
            plan,
            d_brief,
            str(path),
            brief_emb,
            entities=entities,
            rabbit_holes=rabbit_holes,
        )
        store.add_finding(
            conn, chosen_seed.text + " :: " + body[:200], brief_emb
        )
        recent_embs.append(brief_emb)

        # Online taste-profile nudge: high-dopamine briefs pull the nearest cluster centroid in.
        if d_brief["total"] > 0.5 and centroids:
            best = taste.best_cluster_index(centroids, brief_emb)
            if best is not None and best < len(clusters):
                new_c = taste.nudge_centroid(np.asarray(centroids[best]), brief_emb)
                store.update_cluster(conn, clusters[best]["id"], new_c)
                centroids[best] = new_c

        suffix = (
            f"  artifact={artifact_meta['bytes_path']}"
            if artifact_meta and artifact_meta.get("bytes_path")
            else ""
        )
        console.print(
            f"[green]  brief:[/] {path}  dopamine={d_brief['total']:.3f}{suffix}"
        )
        n_done += 1

    entries = store.list_journal(conn, limit=200)
    journal.write_index(entries)
    journal.write_today_notebook(entries)

    session_entries = entries[:n_done]
    top = sorted(
        session_entries,
        key=lambda e: e.get("dopamine_total") or 0.0,
        reverse=True,
    )[:3]

    console.print(
        f"\n[bold]Wrote {n_done} brief(s).[/] Top dopamine from this session:"
    )
    for e in top:
        console.print(
            f"  - {e['dopamine_total']:.3f}  [{e['seed_source']}] "
            f"{e['seed']} -> {e['path']}"
        )
    console.print(f"\nMorning rollup: [dim]journal/today.md[/]")


def _pick_seed(
    clusters: list[dict],
    llm,
    rng: random.Random,
    recent: list[dict],
) -> seeds.Seed:
    """Sample one of the seed strategies according to a fixed mix.

    All cluster-consuming generators receive the *full* cluster dicts (with `meta.theme`
    and `meta.subtopics`) — never the raw `label` fallback that used to leak URLs.
    """
    pick = rng.random()
    if pick < 0.35 or not clusters:
        return seeds.cold_start(clusters, rng)
    if pick < 0.65:
        return seeds.cluster_sample(clusters, llm, rng)
    if pick < 0.85:
        return seeds.cross_pollinate(clusters, llm, rng)
    return seeds.drift(recent, llm, rng)


def _label_for_top_cluster(
    clusters: list[dict], centroids: list[np.ndarray], brief_emb: np.ndarray
) -> str:
    """Return the human-readable label of the cluster nearest to a brief's embedding."""
    if not centroids or not clusters:
        return ""
    idx = taste.best_cluster_index(centroids, brief_emb) or 0
    if idx >= len(clusters):
        return ""
    return clusters[idx].get("label") or ""


def _maybe_generate(
    cluster_label: str,
    prompt_seed: str,
    brief_body: str,
    enabled: set[str],
    dry_run: bool,
    verbose: bool,
) -> Optional[dict]:
    """Pick a generator for the cluster label and produce one artifact, if a backend is available."""
    candidates = gens.pick_for_cluster(cluster_label, dry_run=dry_run)
    candidates = [g for g in candidates if g.modality in enabled]
    if not candidates:
        if verbose:
            console.print(
                f"[yellow]  generators: no available backend for label '{cluster_label}'[/]"
            )
        return None
    gen = candidates[0]
    # Use the seed + first sentence of the brief as a more visual prompt.
    visual_prompt = (prompt_seed + ". " + brief_body[:200]).strip()
    try:
        artifact = gen.generate(visual_prompt)
    except Exception as e:
        if verbose:
            console.print(f"[yellow]  generator {gen.name}: {e}[/]")
        return None
    return _artifact_to_dict(artifact)


def _artifact_to_dict(artifact) -> dict:
    """Serialize an Artifact dataclass into a JSON-safe dict for frontmatter."""
    return {
        "modality": artifact.modality,
        "prompt": artifact.prompt,
        "bytes_path": str(artifact.bytes_path) if artifact.bytes_path else None,
        "url": artifact.url,
        "mime": artifact.mime,
        "generator": artifact.generator,
        "seconds": artifact.seconds,
        "meta": artifact.meta,
    }


if __name__ == "__main__":
    typer.run(main)
