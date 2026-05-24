"""Best-first tree-search wander mode (v0.2): frontier, pruning, UCB, backprop, mutations.

This module is consumed by `wander.py`. v0.1.x flat-mode behavior is unaffected; nothing
in `explore.py` imports from here.

Design:
  - `Frontier` is a heap-backed priority queue with score-update tombstoning.
  - `Pruner` is a small dataclass of three knobs (margin, min_absolute, similarity).
  - `ucb_bonus` is standard UCB1; `visit_count` defaults to 1 so this is meaningful even
    when nodes are popped at most once (v0.2's loop never re-pops).
  - `backprop` walks the parent chain and lifts each ancestor's `subtree_score` to the
    decay-discounted child score — so a deep promising finding rescues a meh-looking root.
  - Mutation operators take the *parent brief dict* (already loaded from `store.get_journal`)
    and return a `Seed` with `source` set to the mutation name. v0.2.1 will read structured
    `entities` / `rabbit_holes` off the same dict; v0.2 mutate_deepen and mutate_branch
    already check those fields and gracefully fall back to LLM-prompting when absent.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from dmn import store
from dmn.seeds import Seed


# ----- Frontier ---------------------------------------------------------------


class Frontier:
    """Priority queue of (score, node_id) keyed best-first.

    Backed by `heapq` (a min-heap), inverting the score to get max-priority. We support
    rescoring a node by tombstoning stale heap entries: `_best_score[node_id]` always holds
    the latest pushed score, and `pop_best` discards entries whose recorded score doesn't
    match.
    """

    def __init__(self) -> None:
        # heap entries: (-score, monotonic_counter, node_id, payload)
        self._heap: list[tuple[float, int, int, dict]] = []
        self._best_score: dict[int, float] = {}
        self._payload: dict[int, dict] = {}
        self._counter: int = 0

    def push(
        self, node_id: int, score: float, payload: Optional[dict] = None
    ) -> None:
        """Insert or rescore a node in the frontier."""
        self._counter += 1
        self._best_score[node_id] = float(score)
        self._payload[int(node_id)] = payload or {}
        heapq.heappush(
            self._heap, (-float(score), self._counter, int(node_id), payload or {})
        )

    def pop_best(
        self, *, total_iters: int = 0, explore_c: float = 0.0
    ) -> Optional[dict]:
        """Pop the best node, or None if the frontier is empty.

        With `explore_c <= 0` (default) this is pure best-first: the highest-scoring open
        node wins. With `explore_c > 0` it adds a UCB1 exploration bonus so the loop
        doesn't tunnel down one greedy branch — shallower / less-committed nodes get a
        premium that decays with depth, restoring the breadth a "wandering mind" wants.
        `total_iters` is the global expansion count (the UCB N term).
        """
        if explore_c and explore_c > 0.0 and self._best_score:
            def priority(nid: int) -> float:
                depth = int(self._payload.get(nid, {}).get("depth", 0))
                # Treat depth as the visit count: a node deep in an exploited branch has
                # effectively been "visited" more, so its exploration bonus shrinks.
                bonus = ucb_bonus({"visit_count": depth + 1}, total_iters, C=explore_c)
                return self._best_score[nid] + bonus

            nid = max(self._best_score, key=priority)
            score = self._best_score.pop(nid)
            payload = self._payload.pop(nid, {})
            return {"id": nid, "score": score, **payload}
        while self._heap:
            neg, _, nid, payload = heapq.heappop(self._heap)
            score = -neg
            current = self._best_score.get(nid)
            if current is not None and abs(current - score) < 1e-9:
                self._best_score.pop(nid, None)
                self._payload.pop(nid, None)
                return {"id": nid, "score": score, **payload}
        return None

    def peek_best(self) -> Optional[dict]:
        """Look at the best node without removing it."""
        while self._heap:
            neg, _, nid, payload = self._heap[0]
            score = -neg
            current = self._best_score.get(nid)
            if current is not None and abs(current - score) < 1e-9:
                return {"id": nid, "score": score, **payload}
            heapq.heappop(self._heap)
        return None

    def size(self) -> int:
        """Number of non-stale nodes still in the frontier."""
        return len(self._best_score)

    def all_open(self) -> list[int]:
        """Snapshot of node ids currently in the frontier."""
        return list(self._best_score.keys())

    def discard(self, node_id: int) -> None:
        """Remove a node from the live frontier; stale heap entries are skipped later."""
        self._best_score.pop(int(node_id), None)
        self._payload.pop(int(node_id), None)


# ----- Pruner -----------------------------------------------------------------


@dataclass
class Pruner:
    """Decides which children survive an expansion.

    - `margin`: a child must score within `parent_score - margin` to survive.
    - `min_absolute`: any brief below this is pruned regardless of parent.
    - `similarity_threshold`: children with cosine > threshold to a sibling are pruned for diversity.
    """

    margin: float = 0.05
    min_absolute: float = 0.15
    similarity_threshold: float = 0.95


def ucb_bonus(node: dict, total_iters: int, C: float = 0.4) -> float:
    """Standard UCB1 exploration bonus: C * sqrt(ln(N_total) / max(1, N_node)).

    `node['visit_count']` is the visit count for this node (defaults to 1). `Frontier.pop_best`
    feeds it `depth + 1` so the bonus decays with how deep (i.e. how committed) a branch is;
    pass `--explore 0` to wander.py to disable and get pure best-first selection.
    """
    n_visits = max(1, int(node.get("visit_count", 1)))
    return C * math.sqrt(math.log(max(2, int(total_iters))) / n_visits)


def backprop(
    conn,
    child_id: int,
    child_score: float,
    decay: float = 0.5,
) -> None:
    """Walk the parent_id chain upward, lifting each ancestor's subtree_score.

    `subtree_score` becomes max(current, child_score * decay**delta) for each ancestor at
    depth `delta` above the child. This is how a promising deep finding rescues a
    meh-looking root.
    """
    visited: set[int] = set()
    cur_id: Optional[int] = int(child_id)
    delta = 0
    while cur_id is not None:
        if cur_id in visited:
            break
        visited.add(cur_id)
        node = store.get_journal(conn, cur_id)
        if node is None:
            break
        propagated = float(child_score) * (float(decay) ** delta)
        existing = node.get("subtree_score")
        existing_v = float(existing) if existing is not None else 0.0
        new_score = max(existing_v, propagated)
        store.update_journal_subtree_score(conn, cur_id, new_score)
        parent_id = node.get("parent_id")
        cur_id = int(parent_id) if parent_id is not None else None
        delta += 1


# ----- Mutation operators -----------------------------------------------------


def mutate_drift(
    parent_brief: dict, llm, rng: Optional[random.Random] = None
) -> Seed:
    """Adjacent question one step sideways from the parent brief's seed."""
    seed_text = parent_brief.get("seed") or ""
    prompt = (
        f"Recent question: '{seed_text}'.\n\n"
        "What's ONE adjacent question, exactly one step sideways? Reply with just the "
        "question, no preamble."
    )
    text = _ask_first_line(
        llm, system="You drift sideways from ideas.", user=prompt
    )
    return Seed(text=text or f"Adjacent to: {seed_text}", source="drift")


def mutate_deepen(
    parent_brief: dict, llm, rng: Optional[random.Random] = None
) -> Seed:
    """Deep-dive on one concrete entity from the parent brief.

    v0.2: prompts the LLM. v0.2.1: samples a salience-weighted entity from
    `parent_brief['entities']`. Either way, gracefully falls back to a generic deep-dive
    prompt if structured entities are absent.
    """
    rng = rng or random.Random()
    entities = parent_brief.get("entities") or []
    rabbit_holes = parent_brief.get("rabbit_holes") or []
    if entities:
        e = _weighted_choice(
            entities, key=lambda x: x.get("salience", 0.5), rng=rng
        )
        name = e.get("name") if isinstance(e, dict) else str(e)
        seed_text = parent_brief.get("seed") or "the broader thread"
        text = (
            f"Tell me something concrete and surprising about {name}, "
            f"in the context of {seed_text}."
        )
        return Seed(text=text, source="deepen", subtopic=name)
    if rabbit_holes:
        rh = rng.choice(rabbit_holes)
        return Seed(text=f"Pursue: {rh}", source="deepen", subtopic=rh)
    return _legacy_llm_deepen(parent_brief, llm)


def _legacy_llm_deepen(parent_brief: dict, llm) -> Seed:
    """v0.2 fallback for `mutate_deepen` when no structured entities are available."""
    seed_text = parent_brief.get("seed") or ""
    prompt = (
        f"This brief was about: '{seed_text}'.\n\n"
        "Pick one concrete entity (a paper, person, concept, place, tool) implied by "
        "the question and ask a deep-dive question about it. Reply with just the "
        "question, no preamble."
    )
    text = _ask_first_line(llm, system="You go deep, not wide.", user=prompt)
    return Seed(text=text or f"Deep-dive: {seed_text}", source="deepen")


def mutate_branch(
    parent_brief: dict, llm, rng: Optional[random.Random] = None
) -> Seed:
    """Pursue a side-thread the parent didn't follow.

    v0.2.1: consumes `parent_brief['rabbit_holes']` when ≥2 are available (avoiding the
    one `mutate_deepen` would have picked). Otherwise re-prompts the LLM.
    """
    rng = rng or random.Random()
    rabbit_holes = parent_brief.get("rabbit_holes") or []
    if len(rabbit_holes) >= 2:
        rh = rng.choice(rabbit_holes[1:])  # skip rh[0] in case deepen picked it
        return Seed(text=f"From the side: {rh}", source="branch", subtopic=rh)
    return _legacy_llm_branch(parent_brief, llm)


def _legacy_llm_branch(parent_brief: dict, llm) -> Seed:
    """v0.2 fallback for `mutate_branch` when no rabbit_holes are available."""
    seed_text = parent_brief.get("seed") or ""
    prompt = (
        f"This brief was about: '{seed_text}'.\n\n"
        "Identify one side-thread the brief didn't follow and pose a research question "
        "about it. Reply with just the question, no preamble."
    )
    text = _ask_first_line(llm, system="You take the road not taken.", user=prompt)
    return Seed(text=text or f"Side of: {seed_text}", source="branch")


def mutate_retool(
    parent_brief: dict,
    llm,
    available_tools: list[str],
) -> tuple[Seed, list[str]]:
    """Re-run the parent's seed but force a different tool palette.

    Returns `(seed, forced_tools)`. The wander loop must respect `forced_tools` instead of
    calling the regular `plan_tools`.
    """
    parent_tools = set(parent_brief.get("tools") or [])
    forced = [t for t in available_tools if t not in parent_tools]
    if not forced:
        # Parent already used everything available — fall back to all tools.
        forced = list(available_tools)
    seed_text = parent_brief.get("seed") or ""
    return Seed(text=seed_text, source="retool", subtopic=None), forced


def mutate_modality_switch(
    parent_brief: dict,
    llm,
    rng: Optional[random.Random],
    eligible_activities: list[str],
) -> tuple[Seed, str]:
    """v0.3: same seed as parent, different activity (e.g. research → code_sketch).

    Returns `(seed, forced_activity)`. The wander loop runs the new activity on the same
    seed text. If `eligible_activities` is empty / has only the parent's activity, falls
    back to a 'drift'-style sideways question with the parent's activity preserved.
    """
    rng = rng or random.Random()
    parent_activity = parent_brief.get("activity") or "research"
    others = [a for a in eligible_activities if a and a != parent_activity]
    if not others:
        # Nothing to switch to — fall through to drift behavior with same activity.
        drift_seed = mutate_drift(parent_brief, llm, rng)
        return drift_seed, parent_activity
    chosen = rng.choice(others)
    seed_text = parent_brief.get("seed") or ""
    return (
        Seed(text=seed_text, source="modality_switch", subtopic=chosen),
        chosen,
    )


# ----- Helpers ---------------------------------------------------------------


def _ask_first_line(llm, system: str, user: str, max_tokens: int = 200) -> str:
    """Prompt the LLM and return the first non-empty stripped line of the response."""
    try:
        resp = llm.complete(system=system, user=user, max_tokens=max_tokens)
        text = (resp.text or "").strip()
    except Exception:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _weighted_choice(
    items: list[Any],
    key,
    rng: Optional[random.Random] = None,
):
    """Sample one item from a list weighted by `key(item)`. Empty list -> raises IndexError."""
    rng = rng or random.Random()
    if not items:
        raise IndexError("cannot choose from an empty list")
    weights = [max(0.0001, float(key(i))) for i in items]
    r = rng.random() * sum(weights)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r < acc:
            return it
    return items[-1]


# ----- Mermaid rendering -----------------------------------------------------


def render_tree_mermaid(nodes: list[dict]) -> str:
    """Render a Mermaid `graph TD` of a wander session's tree, color-coded by status/activity.

    Nodes labelled with id + truncated seed + dopamine + activity. Click handlers point
    to the brief file path. Class definitions: pruned (gray), leaf (blue), and one class
    per activity (research / code_sketch / app_idea / algorithm_explore / ml_experiment /
    image_riff / music_riff / video_riff / mood_journal). v0.3 favors activity color
    over score buckets — score is still on each label.
    """
    if not nodes:
        return "```mermaid\ngraph TD\n  empty[\"empty tree\"]\n```\n"
    lines = ["```mermaid", "graph TD"]
    node_ids = {int(n["id"]) for n in nodes}
    for n in nodes:
        nid = int(n["id"])
        seed = (n.get("seed") or "").replace('"', "'").replace("\n", " ")[:40]
        score = n.get("dopamine_total") or 0.0
        status = n.get("status") or "open"
        activity = n.get("activity") or "research"
        cls = _node_class(status, float(score), activity)
        label = f"#{nid} {seed}<br/>d={float(score):.2f} · {activity}"
        lines.append(f'  n{nid}["{label}"]:::{cls}')
        path = n.get("path") or ""
        if path:
            safe_path = path.replace('"', '\\"')
            lines.append(f'  click n{nid} "{safe_path}"')
    for n in nodes:
        pid = n.get("parent_id")
        if pid is not None and int(pid) in node_ids:
            lines.append(f"  n{int(pid)} --> n{int(n['id'])}")
    lines += [
        "  classDef pruned fill:#cccccc,color:#666",
        "  classDef leaf fill:#a0c4ff,color:#000",
        "  classDef research fill:#c2e7ff,color:#000",
        "  classDef code_sketch fill:#9be08a,color:#000",
        "  classDef app_idea fill:#fff3b0,color:#000",
        "  classDef algorithm_explore fill:#bee3b4,color:#000",
        "  classDef ml_experiment fill:#ffd6a5,color:#000",
        "  classDef image_riff fill:#fcc2d7,color:#000",
        "  classDef music_riff fill:#dbb3ff,color:#000",
        "  classDef video_riff fill:#a4c8ff,color:#000",
        "  classDef mood_journal fill:#fde2c8,color:#000",
    ]
    lines.append("```")
    return "\n".join(lines) + "\n"


def _node_class(status: str, score: float, activity: str = "research") -> str:
    """Map (status, dopamine, activity) to a Mermaid classDef name.

    Status takes precedence (pruned/leaf are visually distinct), then activity name.
    """
    if status == "pruned":
        return "pruned"
    if status == "leaf":
        return "leaf"
    # Restrict to known activity names so unknown values can't break the Mermaid markup.
    known = {
        "research",
        "code_sketch",
        "app_idea",
        "algorithm_explore",
        "ml_experiment",
        "image_riff",
        "music_riff",
        "video_riff",
        "mood_journal",
    }
    return activity if activity in known else "research"


def render_entity_cooccurrence_mermaid(nodes: list[dict], top_n: int = 30) -> str:
    """v0.2.1: Mermaid entity co-occurrence graph across a wander session.

    Aggregates `entities` lists from `nodes`, picks the top-N by total salience-weight,
    and emits a `graph LR` where edges connect entities co-occurring in the same brief.
    Returns the empty string if no node carries entity data.
    """
    # 1. aggregate per-node entity sets + global salience-weight totals
    node_entity_sets: list[set[str]] = []
    salience_totals: dict[str, float] = {}
    entity_types: dict[str, str] = {}
    for n in nodes:
        ents = n.get("entities") or []
        names: set[str] = set()
        for e in ents:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            try:
                salience = float(e.get("salience", 0.5))
            except (TypeError, ValueError):
                salience = 0.5
            salience_totals[name] = salience_totals.get(name, 0.0) + salience
            entity_types.setdefault(name, e.get("type") or "concept")
            names.add(name)
        node_entity_sets.append(names)

    if not salience_totals:
        return ""

    # 2. pick top-N entities by total salience
    top = sorted(salience_totals.items(), key=lambda kv: -kv[1])[:top_n]
    keep = {name for name, _ in top}
    if not keep:
        return ""

    # 3. compute pairwise co-occurrence (count of briefs both entities appear in)
    cooc: dict[tuple[str, str], int] = {}
    for names in node_entity_sets:
        kept = sorted(n for n in names if n in keep)
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                key = (kept[i], kept[j])
                cooc[key] = cooc.get(key, 0) + 1

    # 4. emit Mermaid (graph LR for entity nets — denser, less tree-like than TD)
    lines = ["```mermaid", "graph LR"]
    name_to_id: dict[str, str] = {}
    for idx, (name, _) in enumerate(top):
        nid = f"e{idx}"
        name_to_id[name] = nid
        safe = name.replace('"', "'").replace("\n", " ")[:40]
        kind = entity_types.get(name, "concept")
        lines.append(f'  {nid}["{safe}<br/><i>{kind}</i>"]')
    for (a, b), count in sorted(cooc.items(), key=lambda kv: -kv[1]):
        if a in name_to_id and b in name_to_id and count >= 1:
            if count > 1:
                lines.append(
                    f'  {name_to_id[a]} ---|"x{count}"| {name_to_id[b]}'
                )
            else:
                lines.append(f"  {name_to_id[a]} --- {name_to_id[b]}")
    lines.append("```")
    return "\n".join(lines) + "\n"
