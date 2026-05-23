"""Markdown journal: write briefs with frontmatter, maintain index.md and today.md."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable

JOURNAL_DIR = Path("journal")


def _slug(s: str, n: int = 60) -> str:
    """Filesystem-safe slug from a free-text seed."""
    s = re.sub(r"[^a-zA-Z0-9\-_ ]", "", s or "").strip().lower().replace(" ", "-")
    return (s[:n] or "brief").strip("-") or "brief"


def write_brief(
    seed: str,
    body: str,
    dopamine: dict,
    seed_source: str,
    tools: list[str],
    seed_subtopic: str | None = None,
    artifact: dict | None = None,
    parent_id: int | None = None,
    mutation: str | None = None,
    depth: int | None = None,
    status: str | None = None,
    activity: str | None = None,
    artifacts: list[dict] | None = None,
    execution: dict | None = None,
    journal_dir: Path | str = JOURNAL_DIR,
) -> Path:
    """Write a markdown brief file with YAML frontmatter; returns the new path.

    `seed_subtopic` (v0.1.2) records which subtopic from the cluster's `meta.subtopics` was
    used to ground the seed prompt, when applicable.

    `artifact` is an optional dict describing a generated media file. When provided, it is
    serialized into the frontmatter and a markdown reference (image embed or audio link)
    is appended to the body.

    v0.2: `parent_id`, `mutation`, `depth`, `status` are tree-search fields. They're
    written to frontmatter only when the caller passes them (None = omit, preserving
    flat-mode briefs byte-identically to v0.1.x).
    """
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    fname = f"{now.strftime('%Y-%m-%d-%H%M%S')}-{_slug(seed)}.md"
    path = out_dir / fname
    fm_parts = [
        "---",
        f"seed: {json.dumps(seed)}",
        f"seed_source: {seed_source}",
    ]
    if seed_subtopic:
        fm_parts.append(f"seed_subtopic: {json.dumps(seed_subtopic)}")
    fm_parts.extend(
        [
            f"tools: [{', '.join(tools)}]",
            f"dopamine: {json.dumps(dopamine)}",
            f"created_at: {now.isoformat()}",
        ]
    )
    if parent_id is not None:
        fm_parts.append(f"parent_id: {int(parent_id)}")
    if mutation is not None:
        fm_parts.append(f"mutation: {mutation}")
    if depth is not None:
        fm_parts.append(f"depth: {int(depth)}")
    if status is not None:
        fm_parts.append(f"status: {status}")
    if activity is not None:
        fm_parts.append(f"activity: {activity}")
    if artifact:
        fm_parts.append(f"artifact: {json.dumps(artifact)}")
    if artifacts:
        # serialize each artifact dict; the activity emitted dicts already
        fm_parts.append(f"artifacts: {json.dumps(artifacts)}")
    if execution is not None:
        # full sandbox result; useful for downstream tooling
        fm_parts.append(f"execution: {json.dumps(execution)}")
    fm_parts.extend(["---", "", ""])
    fm = "\n".join(fm_parts)
    body_text = (body or "").rstrip()
    if artifact:
        body_text = body_text + "\n\n" + _render_artifact_md(artifact)
    if artifacts:
        # Activities pass dicts; embed each at the foot of the body.
        embed_lines: list[str] = []
        for a in artifacts:
            md = _render_artifact_md(a)
            if md:
                embed_lines.append(md)
        if embed_lines:
            body_text = body_text + "\n\n" + "\n\n".join(embed_lines)
    path.write_text(fm + body_text + "\n")
    return path


def _render_artifact_md(artifact: dict) -> str:
    """Return the markdown snippet to embed an artifact at the bottom of a brief."""
    modality = artifact.get("modality", "")
    bp = artifact.get("bytes_path") or ""
    url = artifact.get("url") or ""
    target = bp or url
    if not target:
        return ""
    if modality == "image":
        return f"![generated]({target})"
    if modality == "music":
        return f"[Listen]({target}) — {artifact.get('seconds', '?')}s"
    if modality == "video":
        return f"[Watch]({target})"
    return f"[Artifact]({target})"


def write_index(
    entries: Iterable[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/index.md sorted by dopamine_total descending.

    v0.2: tree-mode briefs (anything with a non-null `mutation` other than 'root') are
    grouped under their root and indented by `depth`. Flat-mode briefs render as before.
    """
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict[int, dict] = {
        int(e["id"]): e for e in entries if e.get("id") is not None
    }
    children_of: dict[int, list[dict]] = {}
    for e in by_id.values():
        pid = e.get("parent_id")
        if pid is not None:
            children_of.setdefault(int(pid), []).append(e)
    flat_entries: list[dict] = []
    roots: list[dict] = []
    for e in by_id.values():
        mutation = e.get("mutation")
        parent_id = e.get("parent_id")
        if mutation in (None, "") and parent_id is None:
            # Pure flat-mode brief (no tree fields at all).
            flat_entries.append(e)
        elif parent_id is None:
            # Tree-mode root.
            roots.append(e)
    flat_entries.sort(
        key=lambda x: x.get("dopamine_total") or 0.0, reverse=True
    )
    roots.sort(key=lambda x: x.get("created_at") or 0.0, reverse=True)

    lines = [
        "# DMN Journal Index",
        "",
        "_Flat-mode briefs sorted by dopamine score; tree-mode roots grouped under their session._",
        "",
    ]
    if flat_entries:
        lines.append("## Flat mode")
        lines.append("")
        for e in flat_entries:
            path = e.get("path", "")
            seed = e.get("seed", "?")
            score = e.get("dopamine_total") or 0.0
            tools = ", ".join(e.get("tools") or [])
            activity = e.get("activity") or "research"
            lines.append(
                f"- **{score:.3f}** _[{activity}]_ — [{seed}]({path}) — _{tools}_"
            )
        lines.append("")
    if roots:
        lines.append("## Tree mode")
        lines.append("")
        for root in roots:
            _emit_tree_entry(root, children_of, lines, depth=0)
        lines.append("")
    out = out_dir / "index.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _emit_tree_entry(
    node: dict, children_of: dict[int, list[dict]], lines: list[str], depth: int
) -> None:
    """Recursively emit a tree node + descendants with depth indentation, dopamine-prefixed."""
    indent = "  " * depth
    path = node.get("path", "")
    seed = node.get("seed", "?")
    score = node.get("dopamine_total") or 0.0
    mutation = node.get("mutation") or "root"
    activity = node.get("activity") or "research"
    status = node.get("status") or "open"
    status_tag = "" if status == "open" else f" _[{status}]_"
    lines.append(
        f"{indent}- **{score:.3f}** _({mutation}/{activity})_ — [{seed}]({path}){status_tag}"
    )
    kids = sorted(
        children_of.get(int(node.get("id")), []),
        key=lambda x: x.get("created_at") or 0.0,
    )
    for k in kids:
        _emit_tree_entry(k, children_of, lines, depth + 1)


def write_tree_md(
    nodes: list[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/tree.md: per-session Mermaid graph + top-N table by subtree_score.

    Called at the end of `wander.py` runs; the entity-graph view (v0.2.1) appends a second
    Mermaid block here when entity tagging produced output.
    """
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from dmn.tree import render_tree_mermaid

    lines = [
        f"# Wander tree — {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"_{len(nodes)} brief(s) in this session._",
        "",
        "## Brief tree",
        "",
        render_tree_mermaid(nodes),
        "",
        "## Top by subtree_score",
        "",
        "| score | mutation | depth | seed |",
        "|------:|:---------|------:|:-----|",
    ]
    by_score = sorted(
        nodes,
        key=lambda n: (n.get("subtree_score") or n.get("dopamine_total") or 0.0),
        reverse=True,
    )
    for n in by_score[:10]:
        score = n.get("subtree_score") or n.get("dopamine_total") or 0.0
        mut = n.get("mutation") or "?"
        d = n.get("depth", 0) or 0
        seed = (n.get("seed") or "").replace("|", "\\|")[:80]
        path = n.get("path") or ""
        seed_link = f"[{seed}]({path})" if path else seed
        lines.append(f"| {score:.3f} | {mut} | {d} | {seed_link} |")

    # v0.2.1 hook: if any node carries entity data, also render an entity co-occurrence graph.
    entity_section = _render_entity_graph_section(nodes)
    if entity_section:
        lines.append("")
        lines.append(entity_section)

    out = out_dir / "tree.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _render_entity_graph_section(nodes: list[dict]) -> str:
    """v0.2.1 placeholder. v0.2 nodes don't carry `entities`, so this returns ''.

    Actual implementation lands when entity tagging is wired into `add_journal` /
    `get_journal`; until then we render nothing rather than emit an empty graph.
    """
    has_entities = any(n.get("entities") for n in nodes)
    if not has_entities:
        return ""
    # v0.2.1 will fill this in.
    from dmn.tree import render_entity_cooccurrence_mermaid  # type: ignore

    return (
        "## Entity co-occurrence\n\n"
        + render_entity_cooccurrence_mermaid(nodes)
    )


def write_today_notebook(
    entries: list[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/today.md: rollup of last-24h briefs + an activity-breakdown block (v0.3)."""
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    cutoff = (dt.datetime.now() - dt.timedelta(hours=24)).timestamp()
    recent = [e for e in entries if (e.get("created_at") or 0) >= cutoff]
    recent.sort(key=lambda e: e.get("dopamine_total") or 0.0, reverse=True)
    lines = [
        f"# Today's Notebook — {today.isoformat()}",
        "",
        f"_{len(recent)} brief(s) from the last 24 hours, top first._",
        "",
    ]
    breakdown = _activity_breakdown(recent)
    if breakdown:
        lines.append("## Activity breakdown")
        lines.append("")
        lines.append("| activity | count | mean dopamine |")
        lines.append("|:---------|------:|--------------:|")
        for name, count, mean in breakdown:
            lines.append(f"| {name} | {count} | {mean:.3f} |")
        lines.append("")
    for e in recent:
        path = e.get("path", "")
        seed = e.get("seed", "?")
        score = e.get("dopamine_total") or 0.0
        activity = e.get("activity") or "research"
        lines.append(
            f"\n## [{seed}]({path}) — dopamine {score:.3f} _[{activity}]_"
        )
        lines.append("")
        body_preview = _read_body_preview(path)
        if body_preview:
            lines.append(body_preview)
            lines.append("")
    out = out_dir / "today.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _activity_breakdown(entries: list[dict]) -> list[tuple[str, int, float]]:
    """Return [(activity, count, mean_dopamine)] sorted by count desc."""
    by: dict[str, list[float]] = {}
    for e in entries:
        name = e.get("activity") or "research"
        by.setdefault(name, []).append(float(e.get("dopamine_total") or 0.0))
    rows = [
        (name, len(scores), sum(scores) / len(scores))
        for name, scores in by.items()
    ]
    rows.sort(key=lambda r: -r[1])
    return rows


def _read_body_preview(path: str, max_lines: int = 6) -> str:
    """Strip frontmatter from a brief and return the first few lines for the rollup."""
    if not path:
        return ""
    try:
        text = Path(path).read_text()
    except Exception:
        return ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2] if len(parts) >= 3 else text
    else:
        body = text
    body_lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    return "\n".join(body_lines[:max_lines])
