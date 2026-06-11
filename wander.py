"""DMN best-first tree-search wander mode (v0.2 / v0.3): builds a tree of mutated seeds.

Mirrors `explore.py`'s structure but runs best-first frontier search over a tree of
mutations (drift / deepen / branch / retool / modality_switch). Flat-mode behavior in
`explore.py` is unaffected — this is a separate top-level entry point.

# rationale: v0.3 — pluggable activities (research / code_sketch / app_idea / ...);
#                   `mutate_modality_switch` keeps the same seed but flips the
#                   activity mid-tree, so a research brief can spawn a code_sketch
#                   child on the same idea.
# rationale: v0.2 — best-first frontier over journal rows; mutation operators in
#                   `dmn.tree`; per-session `journal/tree.md` Mermaid; subtree_score
#                   backprop on completion; `--resume` continues from the live frontier.
"""
from __future__ import annotations

import random
import sys
import time
import uuid
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
from dmn import html_journal, journal, portability, seeds, store, taste, tree
from dmn import report as report_mod
from dmn.activities import ActivityContext, ActivityResult
from dmn import llm as llm_mod
from dmn.llm import get_llm
from dmn.loop import available_tools_for, install_graceful_sigint
from dmn.activities.riff_prompts import load_parent_brief_md, make_legacy_media_prompt
from dmn.sandbox import normalize_mode

console = Console()

BANNER = r"""
   ___  __  __  _  _
  |   \|  \/  || \| |   default-mode-network
  | |) | |\/| || .` |   tree-search wander... v{ver}
  |___/|_|  |_||_|\_|
""".strip("\n")


def main(
    minutes: float = typer.Option(
        12.0,
        "--minutes",
        "-m",
        help="Wall-clock budget for the whole tree.",
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Hard cap on briefs across the whole tree (alternative to --minutes).",
    ),
    max_depth: int = typer.Option(
        4, "--max-depth", help="Maximum tree depth before a branch becomes a leaf."
    ),
    children_per_expansion: int = typer.Option(
        3,
        "--children-per-expansion",
        "-k",
        help="How many mutations to attempt per expanded node.",
    ),
    root_count: int = typer.Option(
        3,
        "--root-count",
        help="Number of initial cluster-seeded roots to spawn.",
    ),
    restart_prob: float = typer.Option(
        0.08,
        "--restart-prob",
        help="Per-iter chance of forcing a fresh root instead of expanding.",
    ),
    margin: float = typer.Option(
        0.05,
        "--margin",
        help="Pruner: a child must score within parent_score - margin to survive.",
    ),
    min_absolute: float = typer.Option(
        0.15,
        "--min-absolute",
        help="Pruner: any brief below this is pruned regardless.",
    ),
    similarity_threshold: float = typer.Option(
        0.95,
        "--similarity-threshold",
        help="Pruner: children with cosine > threshold to a sibling are pruned (diversity).",
    ),
    seed_text: Optional[str] = typer.Option(
        None,
        "--seed",
        help="Single explicit seed for the first root.",
    ),
    activities: str = typer.Option(
        "research",
        "--activities",
        help="Comma-separated activities to enable (default: research).",
    ),
    activity_mix: Optional[str] = typer.Option(
        None,
        "--activity-mix",
        help="Weighted activity mix, e.g. 'research:5,code_sketch:2'. Overrides --activities.",
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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Stub LLM and only no-auth tools."
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Legacy v0.1.1 generative-media artifacts (research activity only).",
    ),
    modalities: str = typer.Option(
        "image,music,video",
        "--modalities",
        help="CSV of modalities to enable when --generate is set.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue from the existing open frontier in SQLite instead of fresh roots.",
    ),
    beam_width: int = typer.Option(
        0,
        "--beam-width",
        help="Keep only the top N open frontier leaves after each expansion (0 disables).",
    ),
    patience: int = typer.Option(
        0,
        "--patience",
        help="Stop/restart after this many expansions without meaningful best-score improvement.",
    ),
    until_dopamine: Optional[float] = typer.Option(
        None,
        "--until-dopamine",
        help="Stop early once the global best dopamine reaches this score.",
    ),
    min_improvement: float = typer.Option(
        0.01,
        "--min-improvement",
        help="Minimum best-score delta that resets --patience.",
    ),
    explore: float = typer.Option(
        0.05,
        "--explore",
        help="UCB exploration weight for frontier selection (0 = pure best-first/greedy).",
    ),
    ground: bool = typer.Option(
        True,
        "--ground/--no-ground",
        help="Consume external references before creation activities riff (code/app/web/algo/ml).",
    ),
    report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Write a NotebookLM-style narrated overview of the session to journal/report.html.",
    ),
    code_budget: str = typer.Option(
        "small",
        "--code-budget",
        help="Budget for code-generating activities: small | medium | large.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a best-first tree-search wander session and write briefs + journal/tree.md."""
    console.print(f"[bold magenta]{BANNER.format(ver=__version__)}[/]")
    rng = random.Random()
    run_id = uuid.uuid4().hex[:12]
    started_at = time.time()

    llm = get_llm(dry_run=dry_run)
    llm_mod.announce_and_preflight(
        llm, console, dry_run=dry_run, iterations=iterations, minutes=minutes
    )
    enabled_modalities = {m.strip() for m in modalities.split(",") if m.strip()}
    if generate:
        console.print(
            f"Generators: [bold]on[/] (modalities: {sorted(enabled_modalities)})"
        )

    if execute and no_execute:
        console.print("[red]--execute and --no-execute conflict — pick one.[/]")
        raise typer.Exit(2)
    sandbox_mode = "none" if no_execute else normalize_mode(sandbox)
    if execute and sandbox_mode == "none":
        console.print(
            "[red]--execute needs a sandbox: drop --sandbox none, or use "
            "--sandbox subprocess / docker.[/]"
        )
        raise typer.Exit(2)
    code_budget = code_budget.lower().strip()
    if code_budget not in {"small", "medium", "large"}:
        console.print("[yellow]unknown --code-budget; using small[/]")
        code_budget = "small"
    mix = (
        acts.parse_mix(activity_mix)
        if activity_mix
        else acts.parse_mix(activities)
    )
    eligible_activity_names = sorted(k for k, w in mix.items() if w > 0 and acts.get(k))
    console.print(
        f"Activity mix: {mix}  · execute={execute} sandbox={sandbox_mode}"
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
        ground=ground,
    )
    store.start_run(
        conn,
        run_id,
        command=" ".join(sys.argv),
        notes=f"activity_mix={activity_mix or activities}; code_budget={code_budget}",
    )

    pruner = tree.Pruner(
        margin=margin,
        min_absolute=min_absolute,
        similarity_threshold=similarity_threshold,
    )
    frontier = tree.Frontier()
    deadline = time.time() + minutes * 60 if minutes else None
    n_done = 0
    stop = install_graceful_sigint(console)
    session_node_ids: list[int] = []
    best_score = 0.0
    stale_expansions = 0
    patience_restarts = 0
    patience_triggered = False
    beam_pruned_count = 0

    # Root seeds cycle through clusters ordered for maximal spread, so a session covers
    # many themes instead of collapsing onto the heaviest cluster.
    root_pool = _diverse_root_clusters(clusters, rng)
    _root_focus_idx = 0

    def _next_root_focus() -> Optional[dict]:
        nonlocal _root_focus_idx
        if not root_pool:
            return None
        c = root_pool[_root_focus_idx % len(root_pool)]
        _root_focus_idx += 1
        return c

    if resume:
        for node in store.list_open_frontier(conn):
            score = (
                node.get("subtree_score")
                if node.get("subtree_score") is not None
                else (node.get("dopamine_total") or 0.0)
            )
            frontier.push(
                int(node["id"]),
                float(score),
                payload={"depth": int(node.get("depth") or 0)},
            )
        console.print(f"Resumed with {frontier.size()} open frontier node(s).")

    # --- spawn initial roots ---------------------------------------------------
    if not resume:
        for i in range(root_count):
            if stop["stop"]:
                break
            if iterations is not None and n_done >= iterations:
                break
            if deadline is not None and time.time() >= deadline:
                break
            root_seed = (
                seeds.Seed(text=seed_text, source="manual")
                if (seed_text and i == 0)
                else _pick_root_seed(clusters, llm, rng, focus_cluster=_next_root_focus())
            )
            child_id, score, _ = _expand_brief(
                conn=conn,
                chosen_seed=root_seed,
                parent_id=None,
                depth=0,
                mutation="root",
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                ctx=ctx,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_activity=None,
                forced_tools=None,
                mix=mix,
                run_id=run_id,
                activity_budget=activity_mix or activities,
            )
            if child_id is None:
                continue
            improvement = max(0.0, score - best_score)
            store.update_journal_last_improvement(conn, child_id, improvement)
            if improvement >= min_improvement:
                best_score = max(best_score, score)
                stale_expansions = 0
            else:
                best_score = max(best_score, score)
            frontier.push(child_id, score, payload={"depth": 0})
            session_node_ids.append(child_id)
            n_done += 1
            beam_pruned_count += _prune_frontier_to_beam(
                conn, frontier, run_id, beam_width
            )
            if until_dopamine is not None and best_score >= until_dopamine:
                break

    # --- best-first expansion --------------------------------------------------
    available = available_tools_for(dry_run)
    multi_modal = len(eligible_activity_names) > 1
    while True:
        if stop["stop"]:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if iterations is not None and n_done >= iterations:
            break
        if until_dopamine is not None and best_score >= until_dopamine:
            console.print(
                f"[green]stopping: best dopamine {best_score:.3f} >= {until_dopamine:.3f}[/]"
            )
            break
        if patience and stale_expansions >= patience:
            patience_triggered = True
            if patience_restarts >= max(1, root_count):
                console.print("[yellow]stopping: patience exhausted after restarts[/]")
                break
            console.print("[yellow]patience triggered; restarting from a fresh root[/]")
            patience_restarts += 1
            stale_expansions = 0
            if stop["stop"]:
                break
            root_seed = _pick_root_seed(clusters, llm, rng, focus_cluster=_next_root_focus())
            child_id, score, _ = _expand_brief(
                conn=conn,
                chosen_seed=root_seed,
                parent_id=None,
                depth=0,
                mutation="root",
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                ctx=ctx,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_activity=None,
                forced_tools=None,
                mix=mix,
                run_id=run_id,
                activity_budget=activity_mix or activities,
            )
            if child_id is not None:
                improvement = max(0.0, score - best_score)
                store.update_journal_last_improvement(conn, child_id, improvement)
                best_score = max(best_score, score)
                frontier.push(child_id, score, payload={"depth": 0})
                session_node_ids.append(child_id)
                n_done += 1
                beam_pruned_count += _prune_frontier_to_beam(
                    conn, frontier, run_id, beam_width
                )
            continue

        if frontier.size() == 0 or rng.random() < restart_prob:
            if stop["stop"] or (iterations is not None and n_done >= iterations):
                break
            console.print("[dim]restarting from fresh cluster-seeded root[/]")
            root_seed = _pick_root_seed(clusters, llm, rng, focus_cluster=_next_root_focus())
            child_id, score, _ = _expand_brief(
                conn=conn,
                chosen_seed=root_seed,
                parent_id=None,
                depth=0,
                mutation="root",
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                ctx=ctx,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_activity=None,
                forced_tools=None,
                mix=mix,
                run_id=run_id,
                activity_budget=activity_mix or activities,
            )
            if child_id is not None:
                improvement = max(0.0, score - best_score)
                store.update_journal_last_improvement(conn, child_id, improvement)
                if improvement >= min_improvement:
                    best_score = max(best_score, score)
                    stale_expansions = 0
                else:
                    best_score = max(best_score, score)
                frontier.push(child_id, score, payload={"depth": 0})
                session_node_ids.append(child_id)
                n_done += 1
                beam_pruned_count += _prune_frontier_to_beam(
                    conn, frontier, run_id, beam_width
                )
            continue

        node_summary = frontier.pop_best(total_iters=max(2, n_done), explore_c=explore)
        if node_summary is None:
            continue
        parent_id = int(node_summary["id"])
        store.increment_journal_visit(conn, parent_id)
        parent_node = store.get_journal(conn, parent_id)
        if parent_node is None:
            continue
        parent_depth = int(parent_node.get("depth") or 0)
        if parent_depth >= max_depth:
            store.update_journal_status(conn, parent_id, "leaf")
            continue

        # Build the per-expansion mutation pool. modality_switch is opt-in (depends on
        # whether the user enabled multiple activities).
        all_ops = ["drift", "deepen", "branch", "retool"]
        if multi_modal:
            all_ops.append("modality_switch")
        ops = rng.sample(all_ops, k=min(children_per_expansion, len(all_ops)))
        children: list[tuple[int, float, np.ndarray]] = []
        expansion_best_before = best_score
        for op in ops:
            if stop["stop"]:
                break
            if iterations is not None and n_done >= iterations:
                break
            if deadline is not None and time.time() >= deadline:
                break
            forced_activity: Optional[str] = None
            if op == "retool":
                child_seed, forced = tree.mutate_retool(parent_node, llm, available)
            elif op == "drift":
                child_seed = tree.mutate_drift(parent_node, llm, rng)
                forced = None
            elif op == "deepen":
                child_seed = tree.mutate_deepen(parent_node, llm, rng)
                forced = None
            elif op == "branch":
                child_seed = tree.mutate_branch(parent_node, llm, rng)
                forced = None
            else:  # modality_switch
                child_seed, forced_activity = tree.mutate_modality_switch(
                    parent_node, llm, rng, eligible_activity_names
                )
                forced = None

            child_id, score, child_emb = _expand_brief(
                conn=conn,
                chosen_seed=child_seed,
                parent_id=parent_id,
                depth=parent_depth + 1,
                mutation=op,
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                ctx=ctx,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_activity=forced_activity,
                forced_tools=forced,
                mix=mix,
                run_id=run_id,
                activity_budget=activity_mix or activities,
            )
            if child_id is None:
                continue
            improvement = max(0.0, score - best_score)
            store.update_journal_last_improvement(conn, child_id, improvement)
            best_score = max(best_score, score)
            children.append((child_id, score, child_emb))
            session_node_ids.append(child_id)
            n_done += 1

        # --- pruning pass ----------------------------------------------------
        parent_dopamine = float(parent_node.get("dopamine_total") or 0.0)
        surviving: list[tuple[int, float, np.ndarray]] = []
        for cid, score, child_emb in children:
            if score < pruner.min_absolute:
                store.update_journal_status(conn, cid, "pruned")
                continue
            if score < parent_dopamine - pruner.margin:
                store.update_journal_status(conn, cid, "pruned")
                continue
            killed = False
            for other_cid, _other_score, other_emb in children:
                if other_cid == cid:
                    continue
                if other_emb is None or child_emb is None:
                    continue
                cs = _cosine(child_emb, other_emb)
                if cs > pruner.similarity_threshold:
                    killed = True
                    break
            if killed:
                store.update_journal_status(conn, cid, "pruned")
                continue
            surviving.append((cid, score, child_emb))
            frontier.push(cid, score, payload={"depth": parent_depth + 1})

        if not surviving:
            store.update_journal_status(conn, parent_id, "leaf")
        store.increment_journal_expanded(conn, parent_id)
        batch_improvement = best_score - expansion_best_before
        if batch_improvement >= min_improvement:
            stale_expansions = 0
        else:
            stale_expansions += 1
        beam_pruned_count += _prune_frontier_to_beam(conn, frontier, run_id, beam_width)

    # --- write outputs ---------------------------------------------------------
    session_nodes = store.list_journal_by_run(conn, run_id, limit=1000)
    journal.write_tree_md(session_nodes)

    entries = store.list_journal(conn, limit=max(500, len(session_node_ids)))
    journal.write_index(entries)
    journal.write_today_notebook(entries)
    html_journal.build_html_journal(entries, session_nodes=session_nodes)

    report_path = None
    if report:
        try:
            report_path = report_mod.build_run_report(conn, run_id, llm=llm)
            if report_path:
                html_journal.write_report_html(report_path)
        except Exception as e:
            console.print(f"[yellow]report generation failed: {e}[/]")

    pruned_count = sum(1 for n in session_nodes if n.get("status") == "pruned")
    leaf_count = sum(1 for n in session_nodes if n.get("status") == "leaf")
    open_count = sum(1 for n in session_nodes if n.get("status") == "open")
    console.print(
        f"\n[bold]Wander complete.[/] {n_done} brief(s) "
        f"({pruned_count} pruned, {leaf_count} leaf, {open_count} open). "
        f"best={best_score:.3f}; beam_pruned={beam_pruned_count}; "
        f"patience_triggered={patience_triggered}."
    )
    session_set = set(session_node_ids)
    top = sorted(
        (e for e in entries if e.get("id") in session_set),
        key=lambda e: e.get("dopamine_total") or 0.0,
        reverse=True,
    )[:3]
    if top:
        console.print("Top dopamine from this session:")
        for e in top:
            console.print(
                f"  - {e['dopamine_total']:.3f}  [{e['seed_source']}/{e.get('activity') or 'research'}] "
                f"{e['seed']} -> {e['path']}"
            )
    console.print(
        "\nBrowse: [dim]journal/index.html[/] · tree [dim]journal/tree.html[/] · "
        "health [dim]journal/dashboard.html[/]"
    )
    if report_path:
        console.print(f"[bold]Report →[/] [dim]journal/{Path(report_path).stem}.html[/] (journal/report.html)")
    store.finish_run(
        conn,
        run_id,
        best_score=best_score,
        brief_count=n_done,
        notes=(
            f"beam_pruned={beam_pruned_count}; patience_triggered={patience_triggered}; "
            f"duration_s={time.time() - started_at:.1f}"
        ),
    )


# ----- Helpers ---------------------------------------------------------------


def _diverse_root_clusters(clusters: list[dict], rng: random.Random) -> list[dict]:
    """Order non-noise clusters for maximal topical spread (farthest-point sampling).

    Mass-weighted sampling sends every root into the dominant cluster, so a whole session
    collapses onto one theme. Cycling roots through this ordering instead makes them span
    the taste profile: start at the heaviest cluster, then repeatedly add the cluster most
    distant (lowest cosine) from those already chosen.
    """
    pool = [c for c in clusters if not c.get("is_noise") and c.get("centroid") is not None]
    if len(pool) <= 1:
        return list(pool) or list(clusters)

    def _mass(c: dict) -> float:
        m = (c.get("meta") or {}).get("cluster_mass")
        return float(m) if m is not None else float(c.get("n_members") or 1)

    def _unit(c: dict) -> np.ndarray:
        v = np.asarray(c["centroid"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    units = {id(c): _unit(c) for c in pool}
    ordered = [max(pool, key=_mass)]
    remaining = [c for c in pool if c is not ordered[0]]
    while remaining:
        nxt = max(
            remaining,
            key=lambda c: min(
                1.0 - float(np.dot(units[id(c)], units[id(s)])) for s in ordered
            ),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def _pick_root_seed(
    clusters: list[dict],
    llm,
    rng: random.Random,
    focus_cluster: Optional[dict] = None,
) -> seeds.Seed:
    """Sample a root seed using cluster-grounded strategies (no drift — drift wants a parent).

    When `focus_cluster` is given, seed from that specific cluster so successive roots span
    distinct themes instead of all converging on the heaviest cluster.
    """
    if focus_cluster is not None:
        pool = [focus_cluster]
        if rng.random() < 0.5:
            return seeds.cold_start(pool, rng)
        return seeds.cluster_sample(pool, llm, rng)
    pick = rng.random()
    if pick < 0.45 or not clusters:
        return seeds.cold_start(clusters, rng)
    if pick < 0.90:
        return seeds.cluster_sample(clusters, llm, rng)
    return seeds.cross_pollinate(clusters, llm, rng)


def _expand_brief(
    *,
    conn,
    chosen_seed,
    parent_id: Optional[int],
    depth: int,
    mutation: str,
    clusters: list[dict],
    centroids: list[np.ndarray],
    recent_embs: list[np.ndarray],
    ctx: ActivityContext,
    rng: random.Random,
    verbose: bool,
    generate: bool,
    enabled_modalities: set[str],
    forced_activity: Optional[str],
    forced_tools: Optional[list[str]],
    mix: dict[str, int],
    run_id: str,
    activity_budget: str,
) -> tuple[Optional[int], float, Optional[np.ndarray]]:
    """Run one node: pick activity, run, score, persist, backprop.

    Returns `(journal_id, dopamine_total, brief_embedding)`. Returns `(None, 0.0, None)`
    when the activity produced no body so the loop can keep going without the row.

    `forced_activity` (set by `mutate_modality_switch`) overrides the mix sampler.
    `forced_tools` (set by `mutate_retool`) bypasses `plan_tools`. Both passed only when
    the relevant activity is `research` — other activities ignore them.
    """
    cluster_label = _nearest_cluster_label(
        clusters, centroids, chosen_seed.text, ctx.embed_fn
    )
    if forced_activity:
        activity = acts.get(forced_activity) or acts.get("research")
    else:
        activity = acts.pick_activity(
            mix, ctx, rng, cluster_label=cluster_label, seed_text=chosen_seed.text
        )
    if activity is None:
        console.print("[red]  no activity available; skipping[/]")
        return None, 0.0, None

    console.print(
        Panel(
            f"[bold]Seed:[/] {chosen_seed.text}\n"
            f"[dim]source: {chosen_seed.source}  depth: {depth}  "
            f"mutation: {mutation}  · activity: {activity.name}[/]",
            border_style="cyan",
        )
    )

    node_started = time.time()
    old_forced_tools = getattr(ctx, "forced_tools", None)
    ctx.forced_tools = forced_tools if activity.name == "research" else None
    ctx.cluster_label = cluster_label
    ctx.parent_brief_md = ""
    if parent_id is not None and activity.name in {
        "image_riff",
        "music_riff",
        "video_riff",
    }:
        parent_row = store.get_journal(conn, parent_id)
        ctx.parent_brief_md = load_parent_brief_md(parent_row)
    try:
        result: ActivityResult = activity.run(chosen_seed, ctx)
    except Exception as e:
        console.print(f"[red]  activity {activity.name} crashed: {e}[/]")
        return None, 0.0, None
    finally:
        ctx.forced_tools = old_forced_tools
        ctx.cluster_label = ""
        ctx.parent_brief_md = ""

    if not (result.body_md or "").strip():
        reason = result.metadata.get("reason", "empty body")
        if result.metadata.get("skipped"):
            console.print(f"[yellow]  research skipped ({reason})[/]")
        else:
            console.print("[yellow]  (empty body, skipping)[/]")
        return None, 0.0, None

    embedding_text = result.embedding_text or chosen_seed.text
    brief_emb = ctx.embed_fn([embedding_text[:1500]])[0]
    fulfillment = float(result.metadata.get("fulfillment", 1.0))
    d_brief = taste.dopamine(
        brief_emb, centroids, recent_embs, rng, fulfillment=fulfillment
    )

    artifact_meta: Optional[dict] = None
    if generate and activity.name == "research":
        artifact_meta = _maybe_generate(
            clusters, centroids, brief_emb,
            chosen_seed.text, result.body_md, enabled_modalities, ctx.dry_run, verbose,
            llm=ctx.llm,
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
        parent_id=parent_id,
        mutation=mutation,
        depth=depth,
        status="open",
        activity=activity.name,
        artifacts=artifacts_dicts or None,
        execution=result.execution,
        fulfillment=result.metadata.get("fulfillment"),
        fulfillment_breakdown=result.metadata.get("fulfillment_breakdown"),
        run_id=run_id,
        visit_count=0,
        expanded_count=0,
        activity_budget=activity_budget,
        cost_seconds=time.time() - node_started,
    )
    node_id = store.add_journal(
        conn,
        chosen_seed.text,
        chosen_seed.source,
        result.metadata.get("tools") or [],
        d_brief,
        str(path),
        brief_emb,
        parent_id=parent_id,
        mutation=mutation,
        depth=depth,
        status="open",
        entities=result.metadata.get("entities"),
        rabbit_holes=result.metadata.get("rabbit_holes"),
        activity=activity.name,
        artifact_paths=[str(a.bytes_path) for a in result.artifacts if a.bytes_path],
        execution_result=result.execution,
        run_id=run_id,
        visit_count=0,
        expanded_count=0,
        activity_budget=activity_budget,
        cost_seconds=time.time() - node_started,
        fulfillment=result.metadata.get("fulfillment"),
        fulfillment_breakdown=result.metadata.get("fulfillment_breakdown"),
        grounding=result.metadata.get("grounding"),
    )
    store.add_finding(
        conn, chosen_seed.text + " :: " + result.body_md[:200], brief_emb
    )
    recent_embs.append(brief_emb)

    # Backprop the freshly-scored child up the ancestor chain.
    tree.backprop(conn, node_id, d_brief["total"])

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
    return node_id, float(d_brief["total"]), brief_emb


def _nearest_cluster_label(
    clusters: list[dict],
    centroids: list[np.ndarray],
    seed_text: str,
    embed_fn,
) -> str:
    """Centroid-nearest cluster label for activity-mix boosting."""
    if not centroids or not clusters:
        return ""
    try:
        seed_emb = embed_fn([seed_text])[0]
    except Exception:
        return ""
    idx = taste.best_cluster_index(centroids, seed_emb) or 0
    if idx >= len(clusters):
        return ""
    return clusters[idx].get("label") or ""


def _maybe_generate(
    clusters: list[dict],
    centroids: list[np.ndarray],
    brief_emb: np.ndarray,
    prompt_seed: str,
    brief_body: str,
    enabled: set[str],
    dry_run: bool,
    verbose: bool,
    llm=None,
) -> Optional[dict]:
    """Legacy v0.1.1 generator path (only triggered by --generate, only on research)."""
    label = ""
    if centroids and clusters:
        idx = taste.best_cluster_index(centroids, brief_emb) or 0
        if idx < len(clusters):
            label = clusters[idx].get("label") or ""
    candidates = gens.pick_for_cluster(label, dry_run=dry_run)
    candidates = [g for g in candidates if g.modality in enabled]
    if not candidates:
        if verbose:
            console.print(
                f"[yellow]  generators: no available backend for label '{label}'[/]"
            )
        return None
    gen = candidates[0]
    media_prompt = make_legacy_media_prompt(
        modality=gen.modality,
        seed_text=prompt_seed,
        brief_body=brief_body,
        cluster_label=label,
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; returns 0.0 if either is degenerate."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _code_timeout(code_budget: str) -> float:
    """Execution timeout by code budget."""
    if code_budget == "large":
        return 120.0
    if code_budget == "medium":
        return 90.0
    return 30.0


def _prune_frontier_to_beam(conn, frontier: tree.Frontier, run_id: str, beam_width: int) -> int:
    """Keep only the top beam_width open leaves for the current run."""
    if beam_width <= 0:
        return 0
    open_ids = frontier.all_open()
    if len(open_ids) <= beam_width:
        return 0
    nodes = []
    for nid in open_ids:
        n = store.get_journal(conn, nid)
        if not n or n.get("run_id") != run_id or n.get("status") != "open":
            continue
        score = n.get("subtree_score")
        if score is None:
            score = n.get("dopamine_total") or 0.0
        nodes.append((float(score), int(nid)))
    nodes.sort(reverse=True)
    keep = {nid for _, nid in nodes[:beam_width]}
    pruned = 0
    for _, nid in nodes[beam_width:]:
        if nid in keep:
            continue
        store.update_journal_status(conn, nid, "pruned")
        frontier.discard(nid)
        pruned += 1
    return pruned


if __name__ == "__main__":
    typer.run(main)
