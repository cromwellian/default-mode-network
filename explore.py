"""DMN exploration loop — the agent-modifiable file. Iterate on this to find better strategies.

# rationale: v0.3   — pluggable activities (research / code_sketch / app_idea / ... );
#                     --activities / --activity-mix select the eligible mix;
#                     --execute + --sandbox enable code-running activities;
#                     default (no flags) preserves v0.2.1 research-only behavior.
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
from dmn import activities as acts
from dmn import embeddings as emb
from dmn import generators as gens
from dmn import html_journal, journal, portability, seeds, store, taste
from dmn.activities import ActivityContext, ActivityResult
from dmn import llm as llm_mod
from dmn.llm import get_llm
from dmn.loop import install_graceful_sigint
from dmn.activities.riff_prompts import make_legacy_media_prompt
from dmn.sandbox import normalize_mode

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
    activities: str = typer.Option(
        "research",
        "--activities",
        help="Comma-separated activities to enable (default: research).",
    ),
    activity_mix: Optional[str] = typer.Option(
        None,
        "--activity-mix",
        help="Weighted activity mix, e.g. 'research:5,code_sketch:2,app_idea:1'. "
        "Overrides --activities when set.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Allow code-generating activities to run their output (default off).",
    ),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Alias for --sandbox none; disables code execution.",
    ),
    sandbox: str = typer.Option(
        "auto",
        "--sandbox",
        help="Execution sandbox: 'auto' | 'docker' | 'subprocess' | 'none'.",
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Also call generative-media tools (image/music/video) per brief (legacy v0.1.1).",
    ),
    modalities: str = typer.Option(
        "image,music,video",
        "--modalities",
        help="CSV of modalities to enable when --generate is set.",
    ),
    code_budget: str = typer.Option(
        "small",
        "--code-budget",
        help="Budget for code-generating activities: small | medium | large.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a wandering-mind research session and write briefs to journal/."""
    console.print(f"[bold magenta]{BANNER.format(ver=__version__)}[/]")
    rng = random.Random()

    llm = get_llm(dry_run=dry_run)
    llm_mod.announce_and_preflight(
        llm, console, dry_run=dry_run, iterations=iterations, minutes=minutes
    )

    enabled_modalities = {m.strip() for m in modalities.split(",") if m.strip()}
    if generate:
        console.print(
            f"Generators: [bold]on[/] (modalities: {sorted(enabled_modalities)})"
        )

    sandbox_mode = "none" if no_execute else normalize_mode(sandbox)
    if execute and sandbox_mode == "none":
        console.print("[yellow]--execute requested but --sandbox none; ignoring --execute[/]")
        execute = False
    code_budget = code_budget.lower().strip()
    if code_budget not in {"small", "medium", "large"}:
        console.print("[yellow]unknown --code-budget; using small[/]")
        code_budget = "small"
    mix = (
        acts.parse_mix(activity_mix)
        if activity_mix
        else acts.parse_mix(activities)
    )
    console.print(f"Activity mix: {mix}  · execute={execute} sandbox={sandbox_mode}")

    conn = store.connect()
    interests = store.list_interests(conn)
    clusters = store.list_clusters(conn)
    if not interests:
        console.print(
            "[red]No taste profile found. Run `uv run prepare.py --interactive` "
            "(or `--dry-run`) first.[/]"
        )
        raise typer.Exit(1)
    mismatch = portability.profile_embedding_mismatch(conn)
    if mismatch:
        console.print(f"[red]{mismatch}[/]")
        raise typer.Exit(1)

    centroids: list[np.ndarray] = [
        c["centroid"] for c in clusters if c["centroid"] is not None
    ]
    recent_findings = store.list_recent_findings(conn, limit=50)
    recent_embs: list[np.ndarray] = [
        r["embedding"] for r in recent_findings if r["embedding"] is not None
    ]

    ctx = ActivityContext(
        llm=llm,
        embed_fn=emb.embed,
        clusters=clusters,
        centroids=centroids,
        recent_embs=recent_embs,
        rng=rng,
        dry_run=dry_run,
        execute=execute,
        sandbox=sandbox_mode,
        verbose=verbose,
        timeout_s=_code_timeout(code_budget),
        code_budget=code_budget,
    )

    deadline = time.time() + minutes * 60 if minutes else None
    n_done = 0
    stop = install_graceful_sigint(console)

    while True:
        if stop["stop"]:
            break
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

        cluster_label = _nearest_cluster_label(clusters, centroids, chosen_seed.text)
        ctx.cluster_label = cluster_label
        ctx.parent_brief_md = ""
        activity = acts.pick_activity(
            mix, ctx, rng, cluster_label=cluster_label, seed_text=chosen_seed.text
        )
        if activity is None:
            console.print("[red]  no activity available; aborting[/]")
            break

        console.print(
            Panel(
                f"[bold]Seed:[/] {chosen_seed.text}\n"
                f"[dim]source: {chosen_seed.source}  · activity: {activity.name}[/]",
                border_style="cyan",
            )
        )

        try:
            result: ActivityResult = activity.run(chosen_seed, ctx)
        except Exception as e:
            console.print(f"[red]  activity {activity.name} crashed: {e}[/]")
            n_done += 1
            continue

        if not (result.body_md or "").strip():
            reason = result.metadata.get("reason", "empty body")
            if result.metadata.get("skipped"):
                console.print(f"[yellow]  research skipped ({reason})[/]")
            else:
                console.print("[yellow]  (empty body, skipping)[/]")
            if not result.metadata.get("skipped"):
                n_done += 1
            continue

        # Score the brief itself for dopamine + persistence.
        embedding_text = result.embedding_text or chosen_seed.text
        brief_emb = emb.embed([embedding_text[:1500]])[0]
        fulfillment = float(result.metadata.get("fulfillment", 1.0))
        d_brief = taste.dopamine(
            brief_emb, centroids, recent_embs, rng, fulfillment=fulfillment
        )

        # Pre-v0.3 image/music/video legacy path; off unless --generate is set.
        artifact_meta: Optional[dict] = None
        if generate and activity.name == "research":
            artifact_meta = _maybe_generate(
                cluster_label, chosen_seed.text, result.body_md,
                enabled_modalities, dry_run, verbose, llm=llm,
            )

        artifacts_dicts = [_artifact_to_dict(a) for a in result.artifacts]
        path = journal.write_brief(
            seed=chosen_seed.text,
            body=result.body_md,
            dopamine=d_brief,
            seed_source=chosen_seed.source,
            seed_subtopic=getattr(chosen_seed, "subtopic", None),
            tools=result.metadata.get("tools") or [],
            artifact=artifact_meta,
            activity=activity.name,
            artifacts=artifacts_dicts or None,
            execution=result.execution,
            fulfillment=result.metadata.get("fulfillment"),
            fulfillment_breakdown=result.metadata.get("fulfillment_breakdown"),
        )
        store.add_journal(
            conn,
            chosen_seed.text,
            chosen_seed.source,
            result.metadata.get("tools") or [],
            d_brief,
            str(path),
            brief_emb,
            entities=result.metadata.get("entities"),
            rabbit_holes=result.metadata.get("rabbit_holes"),
            activity=activity.name,
            artifact_paths=[str(a.bytes_path) for a in result.artifacts if a.bytes_path],
            execution_result=result.execution,
            fulfillment=result.metadata.get("fulfillment"),
            fulfillment_breakdown=result.metadata.get("fulfillment_breakdown"),
            grounding=result.metadata.get("grounding"),
        )
        store.add_finding(
            conn, chosen_seed.text + " :: " + result.body_md[:200], brief_emb
        )
        recent_embs.append(brief_emb)

        # Online taste-profile nudge: high-dopamine briefs pull the nearest cluster centroid in.
        if d_brief["total"] > 0.5 and centroids:
            best = taste.best_cluster_index(centroids, brief_emb)
            if best is not None and best < len(clusters):
                new_c = taste.nudge_centroid(np.asarray(centroids[best]), brief_emb)
                store.update_cluster(conn, clusters[best]["id"], new_c)
                centroids[best] = new_c

        suffix = ""
        if artifact_meta and artifact_meta.get("bytes_path"):
            suffix = f"  artifact={artifact_meta['bytes_path']}"
        elif result.artifacts:
            suffix = f"  artifacts={len(result.artifacts)}"
        if result.execution is not None:
            suffix += f"  exec={result.execution.get('exit_code')}"
        console.print(
            f"[green]  brief:[/] {path}  dopamine={d_brief['total']:.3f}  "
            f"activity={activity.name}{suffix}"
        )
        n_done += 1

    entries = store.list_journal(conn, limit=200)
    journal.write_index(entries)
    journal.write_today_notebook(entries)
    html_journal.build_html_journal(entries)

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
            f"  - {e['dopamine_total']:.3f}  [{e['seed_source']}/{e.get('activity') or 'research'}] "
            f"{e['seed']} -> {e['path']}"
        )
    console.print(
        "\nBrowse: [dim]journal/index.html[/] · today [dim]journal/today.html[/] · "
        "health [dim]journal/dashboard.html[/]"
    )


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


def _nearest_cluster_label(
    clusters: list[dict], centroids: list[np.ndarray], seed_text: str
) -> str:
    """Return the label of the cluster whose centroid best matches the seed text.

    Used to nudge `acts.pick_activity` toward modality-appropriate activities (e.g. a
    music cluster lightly boosts `music_riff`).
    """
    if not centroids or not clusters:
        return ""
    try:
        seed_emb = emb.embed([seed_text])[0]
    except Exception:
        return ""
    idx = taste.best_cluster_index(centroids, seed_emb) or 0
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
    llm=None,
) -> Optional[dict]:
    """Legacy v0.1.1 generator pipeline (only triggered by --generate). Activities are preferred."""
    candidates = gens.pick_for_cluster(cluster_label, dry_run=dry_run)
    candidates = [g for g in candidates if g.modality in enabled]
    if not candidates:
        if verbose:
            console.print(
                f"[yellow]  generators: no available backend for label '{cluster_label}'[/]"
            )
        return None
    gen = candidates[0]
    media_prompt = make_legacy_media_prompt(
        modality=gen.modality,
        seed_text=prompt_seed,
        brief_body=brief_body,
        cluster_label=cluster_label,
        llm=llm,
    )
    try:
        artifact = gen.generate(media_prompt)
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


def _code_timeout(code_budget: str) -> float:
    if code_budget == "large":
        return 120.0
    if code_budget == "medium":
        return 90.0
    return 30.0


if __name__ == "__main__":
    typer.run(main)
