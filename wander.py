"""DMN best-first tree-search wander mode (v0.2): builds a tree of mutated seeds.

Mirrors `explore.py`'s structure but runs best-first frontier search over a tree of
mutations (drift / deepen / branch / retool). Flat-mode behavior in `explore.py` is
unaffected — this is a separate top-level entry point.

# rationale: v0.2 — best-first frontier over journal rows; mutation operators in
#                   `dmn.tree`; per-session `journal/tree.md` Mermaid; subtree_score
#                   backprop on completion; `--resume` continues from the live frontier.
"""
from __future__ import annotations

import random
import time
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
from dmn import journal, seeds, store, taste, tree
from dmn.llm import get_llm
from dmn.loop import (
    SYNTHESIS_SYSTEM,
    available_tools_for,
    execute_tools,
    parse_synthesis,
    plan_tools,
)
from dmn.tools import ResearchItem

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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Stub LLM and only no-auth tools."
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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue from the existing open frontier in SQLite instead of fresh roots.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a best-first tree-search wander session and write briefs + journal/tree.md."""
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

    pruner = tree.Pruner(
        margin=margin,
        min_absolute=min_absolute,
        similarity_threshold=similarity_threshold,
    )
    frontier = tree.Frontier()
    deadline = time.time() + minutes * 60 if minutes else None
    n_done = 0
    session_node_ids: list[int] = []

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
            if iterations is not None and n_done >= iterations:
                break
            if deadline is not None and time.time() >= deadline:
                break
            root_seed = (
                seeds.Seed(text=seed_text, source="manual")
                if (seed_text and i == 0)
                else _pick_root_seed(clusters, llm, rng)
            )
            child_id, score, child_emb = _expand_brief(
                conn=conn,
                chosen_seed=root_seed,
                parent_id=None,
                depth=0,
                mutation="root",
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                llm=llm,
                dry_run=dry_run,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_tools=None,
            )
            if child_id is None:
                continue
            frontier.push(child_id, score, payload={"depth": 0})
            session_node_ids.append(child_id)
            n_done += 1

    # --- best-first expansion --------------------------------------------------
    available = available_tools_for(dry_run)
    while True:
        if deadline is not None and time.time() >= deadline:
            break
        if iterations is not None and n_done >= iterations:
            break

        if frontier.size() == 0 or rng.random() < restart_prob:
            if iterations is not None and n_done >= iterations:
                break
            console.print("[dim]restarting from fresh cluster-seeded root[/]")
            root_seed = _pick_root_seed(clusters, llm, rng)
            child_id, score, _ = _expand_brief(
                conn=conn,
                chosen_seed=root_seed,
                parent_id=None,
                depth=0,
                mutation="root",
                clusters=clusters,
                centroids=centroids,
                recent_embs=recent_embs,
                llm=llm,
                dry_run=dry_run,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_tools=None,
            )
            if child_id is not None:
                frontier.push(child_id, score, payload={"depth": 0})
                session_node_ids.append(child_id)
                n_done += 1
            continue

        node_summary = frontier.pop_best()
        if node_summary is None:
            continue
        parent_id = int(node_summary["id"])
        parent_node = store.get_journal(conn, parent_id)
        if parent_node is None:
            continue
        parent_depth = int(parent_node.get("depth") or 0)
        if parent_depth >= max_depth:
            store.update_journal_status(conn, parent_id, "leaf")
            continue

        all_ops = ["drift", "deepen", "branch", "retool"]
        ops = rng.sample(all_ops, k=min(children_per_expansion, len(all_ops)))
        children: list[tuple[int, float, np.ndarray]] = []
        for op in ops:
            if iterations is not None and n_done >= iterations:
                break
            if deadline is not None and time.time() >= deadline:
                break
            if op == "retool":
                child_seed, forced = tree.mutate_retool(parent_node, llm, available)
            elif op == "drift":
                child_seed = tree.mutate_drift(parent_node, llm, rng)
                forced = None
            elif op == "deepen":
                child_seed = tree.mutate_deepen(parent_node, llm, rng)
                forced = None
            else:  # branch
                child_seed = tree.mutate_branch(parent_node, llm, rng)
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
                llm=llm,
                dry_run=dry_run,
                rng=rng,
                verbose=verbose,
                generate=generate,
                enabled_modalities=enabled_modalities,
                forced_tools=forced,
            )
            if child_id is None:
                continue
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

    # --- write outputs ---------------------------------------------------------
    session_nodes = [
        n for n in (store.get_journal(conn, nid) for nid in session_node_ids) if n
    ]
    journal.write_tree_md(session_nodes)

    entries = store.list_journal(conn, limit=500)
    journal.write_index(entries)
    journal.write_today_notebook(entries)

    pruned_count = sum(1 for n in session_nodes if n.get("status") == "pruned")
    leaf_count = sum(1 for n in session_nodes if n.get("status") == "leaf")
    open_count = sum(1 for n in session_nodes if n.get("status") == "open")
    console.print(
        f"\n[bold]Wander complete.[/] {n_done} brief(s) "
        f"({pruned_count} pruned, {leaf_count} leaf, {open_count} open). "
        f"Tree → [dim]journal/tree.md[/]"
    )


# ----- Helpers ---------------------------------------------------------------


def _pick_root_seed(clusters: list[dict], llm, rng: random.Random) -> seeds.Seed:
    """Sample a root seed using cluster-grounded strategies (no drift — drift wants a parent)."""
    pick = rng.random()
    if pick < 0.4 or not clusters:
        return seeds.cold_start(clusters, rng)
    if pick < 0.8:
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
    llm,
    dry_run: bool,
    rng: random.Random,
    verbose: bool,
    generate: bool,
    enabled_modalities: set[str],
    forced_tools: Optional[list[str]],
) -> tuple[Optional[int], float, Optional[np.ndarray]]:
    """Run one node: plan tools, execute, synthesize, score, persist, backprop.

    Returns `(journal_id, dopamine_total, brief_embedding)`. Returns `(None, 0.0, None)`
    when the tools produced no results so the loop can keep going without the row.
    """
    console.print(
        Panel(
            f"[bold]Seed:[/] {chosen_seed.text}\n"
            f"[dim]source: {chosen_seed.source}  depth: {depth}  mutation: {mutation}[/]",
            border_style="cyan",
        )
    )
    plan = (
        list(forced_tools)
        if forced_tools
        else plan_tools(chosen_seed.text, llm, dry_run, rng)
    )
    console.print(f"  using tools: {plan or '(none available)'}")

    items = execute_tools(
        plan,
        chosen_seed.text,
        verbose=verbose,
        log=console.print if verbose else None,
    )
    if not items:
        console.print("[yellow]  (no results, skipping)[/]")
        return None, 0.0, None

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

    bullets: list[str] = [
        f"- [{it.source}] **{it.title}** — {it.summary[:240]}\n"
        f"  {it.url}\n"
        f"  dopamine: {d}"
        for _, d, it in top_k
    ]
    gathered = "\n".join(bullets)
    prompt = (
        f"Seed question: {chosen_seed.text}\n\n"
        f"Gathered findings:\n{gathered}\n\n"
        "Write the brief now."
    )
    try:
        resp = llm.complete(system=SYNTHESIS_SYSTEM, user=prompt, max_tokens=800)
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
        artifact_meta = _maybe_generate(
            clusters, centroids, brief_emb,
            chosen_seed.text, body, enabled_modalities, dry_run, verbose,
        )

    path = journal.write_brief(
        seed=chosen_seed.text,
        body=body,
        dopamine=d_brief,
        seed_source=chosen_seed.source,
        seed_subtopic=getattr(chosen_seed, "subtopic", None),
        tools=plan,
        artifact=artifact_meta,
        parent_id=parent_id,
        mutation=mutation,
        depth=depth,
        status="open",
    )
    node_id = store.add_journal(
        conn,
        chosen_seed.text,
        chosen_seed.source,
        plan,
        d_brief,
        str(path),
        brief_emb,
        parent_id=parent_id,
        mutation=mutation,
        depth=depth,
        status="open",
        entities=entities,
        rabbit_holes=rabbit_holes,
    )
    store.add_finding(
        conn, chosen_seed.text + " :: " + body[:200], brief_emb
    )
    recent_embs.append(brief_emb)

    # Backprop the freshly-scored child up the ancestor chain.
    tree.backprop(conn, node_id, d_brief["total"])

    suffix = (
        f"  artifact={artifact_meta['bytes_path']}"
        if artifact_meta and artifact_meta.get("bytes_path")
        else ""
    )
    console.print(
        f"[green]  brief:[/] {path}  dopamine={d_brief['total']:.3f}{suffix}"
    )
    return node_id, float(d_brief["total"]), brief_emb


def _maybe_generate(
    clusters: list[dict],
    centroids: list[np.ndarray],
    brief_emb: np.ndarray,
    prompt_seed: str,
    brief_body: str,
    enabled: set[str],
    dry_run: bool,
    verbose: bool,
) -> Optional[dict]:
    """Pick a generator for the cluster nearest to a brief and produce one artifact, if any."""
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
    visual_prompt = (prompt_seed + ". " + brief_body[:200]).strip()
    try:
        artifact = gen.generate(visual_prompt)
    except Exception as e:
        if verbose:
            console.print(f"[yellow]  generator {gen.name}: {e}[/]")
        return None
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


if __name__ == "__main__":
    typer.run(main)
