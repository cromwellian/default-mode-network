"""HTML journal: human-readable views alongside markdown briefs."""
from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

from dmn.journal import JOURNAL_DIR, _activity_breakdown
from dmn.paths import artifact_href
from dmn.tree import render_tree_mermaid

# Activity chip colors — aligned with dmn/tree.py Mermaid classDefs.
ACTIVITY_COLORS: dict[str, str] = {
    "research": "#5b9fd4",
    "code_sketch": "#6bbf59",
    "app_idea": "#d4b84a",
    "algorithm_explore": "#7ec87a",
    "ml_experiment": "#e0a060",
    "image_riff": "#e08aaa",
    "music_riff": "#b088e0",
    "video_riff": "#6898d4",
    "mood_journal": "#d4a888",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');

:root {
  --bg: #141210;
  --bg-elev: #1c1916;
  --bg-card: #221e1a;
  --border: #3a342c;
  --text: #e8e2d8;
  --text-dim: #9a9288;
  --accent: #c9a227;
  --accent-dim: #8a7020;
  --link: #7eb8da;
  --code-bg: #0f0d0b;
  --radius: 6px;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  --serif: 'Literata', Georgia, serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.6;
  min-height: 100vh;
}

a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }

.site-header {
  border-bottom: 1px solid var(--border);
  padding: 1.25rem 2rem;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  background: var(--bg-elev);
}

.site-title {
  font-family: var(--serif);
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.site-title span { color: var(--accent); }

nav.site-nav { display: flex; gap: 1.25rem; font-size: 0.9rem; }
nav.site-nav a { color: var(--text-dim); }
nav.site-nav a.active, nav.site-nav a:hover { color: var(--text); }

main { max-width: 52rem; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
main.wide { max-width: 72rem; }

.meta-line {
  color: var(--text-dim);
  font-size: 0.875rem;
  margin-bottom: 1.5rem;
}

.chip {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  color: #111;
  vertical-align: middle;
}

.chip-muted {
  background: var(--border);
  color: var(--text-dim);
  text-transform: none;
  letter-spacing: 0;
}

.dopamine-wrap { display: flex; align-items: center; gap: 0.6rem; min-width: 7rem; }
.dopamine-score {
  font-family: var(--mono);
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--accent);
  min-width: 3.2rem;
  text-align: right;
}
.dopamine-bar {
  flex: 1;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
  min-width: 3rem;
}
.dopamine-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-dim), var(--accent));
  border-radius: 2px;
}

.brief-list { list-style: none; }
.brief-item {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
  margin-bottom: 0.75rem;
  transition: border-color 0.15s;
}
.brief-item:hover { border-color: var(--accent-dim); }
.brief-item a.item-link {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 1rem;
  padding: 1rem 1.15rem;
  align-items: start;
  color: inherit;
  text-decoration: none;
}
.brief-item a.item-link:hover { text-decoration: none; }
.brief-seed {
  font-family: var(--serif);
  font-size: 0.95rem;
  line-height: 1.45;
  margin-bottom: 0.35rem;
}
.brief-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.brief-tools { color: var(--text-dim); font-size: 0.78rem; }

.section-title {
  font-family: var(--serif);
  font-size: 1.1rem;
  margin: 2rem 0 1rem;
  color: var(--text-dim);
  font-weight: 600;
}

.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  margin: 1rem 0 2rem;
}
.stats-table th, .stats-table td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.75rem;
  text-align: left;
}
.stats-table th { background: var(--bg-elev); color: var(--text-dim); font-weight: 500; }
.stats-table td.num { text-align: right; font-family: var(--mono); }

/* Brief page prose */
article.brief-body {
  font-family: var(--serif);
  font-size: 1.05rem;
  line-height: 1.75;
}
article.brief-body h1, article.brief-body h2, article.brief-body h3 {
  font-family: var(--sans);
  margin: 1.75rem 0 0.75rem;
  line-height: 1.3;
}
article.brief-body h1 { font-size: 1.5rem; }
article.brief-body h2 { font-size: 1.25rem; color: var(--accent); }
article.brief-body p { margin-bottom: 1rem; }
article.brief-body ul, article.brief-body ol { margin: 0 0 1rem 1.25rem; }
article.brief-body li { margin-bottom: 0.35rem; }
article.brief-body blockquote {
  border-left: 3px solid var(--accent-dim);
  padding-left: 1rem;
  color: var(--text-dim);
  margin: 1rem 0;
}
article.brief-body img {
  max-width: 100%;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  margin: 1rem 0;
}
article.brief-body audio { width: 100%; margin: 1rem 0; }
article.brief-body pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  overflow-x: auto;
  margin: 1rem 0;
  font-family: var(--mono);
  font-size: 0.82rem;
  line-height: 1.5;
}
article.brief-body code {
  font-family: var(--mono);
  font-size: 0.88em;
  background: var(--code-bg);
  padding: 0.1em 0.35em;
  border-radius: 3px;
}
article.brief-body pre code { background: none; padding: 0; }
article.brief-body table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--sans);
  font-size: 0.9rem;
  margin: 1rem 0;
}
article.brief-body th, article.brief-body td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.65rem;
}
article.brief-body th { background: var(--bg-elev); }

.brief-header { margin-bottom: 2rem; }
.brief-header h1 {
  font-family: var(--serif);
  font-size: 1.45rem;
  line-height: 1.35;
  margin-bottom: 1rem;
}
.dopamine-breakdown {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.78rem;
  font-family: var(--mono);
  color: var(--text-dim);
  margin-top: 0.75rem;
}

.tree-container {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  margin: 1.5rem 0;
  overflow-x: auto;
}
.tree-nested { list-style: none; padding-left: 0; }
.tree-nested ul { list-style: none; padding-left: 1.25rem; border-left: 1px solid var(--border); margin-left: 0.5rem; }
.tree-node { padding: 0.35rem 0; font-size: 0.875rem; }
.tree-node .score { font-family: var(--mono); color: var(--accent); font-size: 0.8rem; }

.mermaid { background: transparent; }
"""

_md = MarkdownIt("commonmark", {"html": True}).enable("table")


def _highlight_code(code: str, lang: str, attrs: str) -> str:
    """Pygments-backed fence renderer for markdown_it."""
    try:
        lexer = get_lexer_by_name(lang or "text", stripall=True)
    except ClassNotFound:
        lexer = TextLexer(stripall=True)
    formatter = HtmlFormatter(
        nowrap=True,
        style="native",
        cssclass="highlight",
    )
    highlighted = highlight(code, lexer, formatter)
    lang_class = f' class="language-{html.escape(lang or "text")}"' if lang else ""
    return f"<pre><code{lang_class}>{highlighted}</code></pre>\n"


_md.options["highlight"] = _highlight_code


def build_html_journal(
    entries: Iterable[dict],
    journal_dir: Path | str = JOURNAL_DIR,
    session_nodes: list[dict] | None = None,
) -> dict[str, Path]:
    """Rebuild all HTML journal pages. Returns paths to index/today/tree."""
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_list = list(entries)

    for e in entry_list:
        md_path = e.get("path")
        if md_path and Path(md_path).exists():
            write_brief_html(Path(md_path), out_dir)

    # Also convert any markdown briefs on disk not in SQLite (stale paths).
    known_md = {e.get("path") for e in entry_list if e.get("path")}
    for md in sorted(out_dir.glob("*.md")):
        if str(md) not in known_md:
            write_brief_html(md, out_dir)

    nodes = session_nodes if session_nodes is not None else _latest_session_nodes(entry_list)
    paths = {
        "index": write_index_html(entry_list, out_dir),
        "today": write_today_html(entry_list, out_dir),
        "tree": write_tree_html(nodes, out_dir),
    }
    return paths


def write_brief_html(md_path: Path, journal_dir: Path | str = JOURNAL_DIR) -> Path:
    """Render one markdown brief to HTML."""
    md_path = Path(md_path)
    out_dir = Path(journal_dir)
    html_path = out_dir / (md_path.stem + ".html")
    fm, body = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
    body_html = _md_to_html(body, md_path)
    seed = fm.get("seed", md_path.stem)
    if isinstance(seed, str) and seed.startswith('"'):
        try:
            seed = json.loads(seed)
        except json.JSONDecodeError:
            pass

    dopamine = fm.get("dopamine") or {}
    if isinstance(dopamine, str):
        try:
            dopamine = json.loads(dopamine)
        except json.JSONDecodeError:
            dopamine = {}
    total = float(dopamine.get("total", 0.0))

    activity = fm.get("activity") or "research"
    mutation = fm.get("mutation")
    depth = fm.get("depth")
    status = fm.get("status")
    created = fm.get("created_at", "")

    tags = [_activity_chip_html(activity)]
    if mutation:
        tags.append(f'<span class="chip chip-muted">{html.escape(str(mutation))}</span>')
    if depth is not None:
        tags.append(f'<span class="chip chip-muted">depth {int(depth)}</span>')
    if status and status != "open":
        tags.append(f'<span class="chip chip-muted">{html.escape(str(status))}</span>')

    breakdown = " · ".join(
        f"{k} {float(v):.2f}"
        for k, v in sorted(dopamine.items())
        if k != "total" and isinstance(v, (int, float))
    )

    page = _page(
        title=str(seed)[:80],
        nav_active="brief",
        body=f"""
<article>
  <header class="brief-header">
    <h1>{html.escape(str(seed))}</h1>
    <div class="brief-tags">{''.join(tags)}</div>
    <div style="margin-top:1rem">{_dopamine_bar_html(total)}</div>
    {f'<div class="dopamine-breakdown">{html.escape(breakdown)}</div>' if breakdown else ''}
    <p class="meta-line" style="margin-top:0.75rem">
      {html.escape(str(created))} · source {html.escape(str(fm.get('seed_source', '?')))}
    </p>
  </header>
  <div class="brief-body">{body_html}</div>
</article>
""",
    )
    html_path.write_text(page, encoding="utf-8")
    return html_path


def write_index_html(
    entries: Iterable[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/index.html sorted by dopamine."""
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_list = sorted(
        list(entries),
        key=lambda x: x.get("dopamine_total") or 0.0,
        reverse=True,
    )
    max_score = max((e.get("dopamine_total") or 0.0 for e in entry_list), default=1.0)
    max_score = max(max_score, 0.001)

    items: list[str] = []
    for e in entry_list:
        md_path = e.get("path") or ""
        href = _md_to_html_href(md_path)
        seed = html.escape(e.get("seed") or "?")
        score = float(e.get("dopamine_total") or 0.0)
        activity = e.get("activity") or "research"
        mutation = e.get("mutation")
        depth = e.get("depth")
        tools = html.escape(", ".join(e.get("tools") or []))

        tags = [_activity_chip_html(activity)]
        if mutation:
            tags.append(
                f'<span class="chip chip-muted">{html.escape(str(mutation))}</span>'
            )
        if depth:
            tags.append(f'<span class="chip chip-muted">d{int(depth)}</span>')

        items.append(
            f"""<li class="brief-item">
  <a class="item-link" href="{html.escape(href)}">
    <div>{_dopamine_bar_html(score, max_score)}</div>
    <div>
      <div class="brief-seed">{seed}</div>
      <div class="brief-tags">{''.join(tags)}</div>
      {f'<div class="brief-tools">{tools}</div>' if tools else ''}
    </div>
  </a>
</li>"""
        )

    body = f"""
<p class="meta-line">{len(entry_list)} brief(s), sorted by dopamine.</p>
<ul class="brief-list">
{''.join(items) if items else '<li class="meta-line">No briefs yet.</li>'}
</ul>
"""
    out = out_dir / "index.html"
    out.write_text(_page("Index", "index", body, wide=True), encoding="utf-8")
    return out


def write_today_html(
    entries: Iterable[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/today.html — last 24h rollup."""
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    cutoff = (dt.datetime.now() - dt.timedelta(hours=24)).timestamp()
    recent = [e for e in entries if (e.get("created_at") or 0) >= cutoff]
    recent.sort(key=lambda e: e.get("dopamine_total") or 0.0, reverse=True)

    breakdown_rows = ""
    for name, count, mean in _activity_breakdown(recent):
        breakdown_rows += (
            f"<tr><td>{html.escape(name)}</td>"
            f'<td class="num">{count}</td>'
            f'<td class="num">{mean:.3f}</td></tr>\n'
        )

    sections: list[str] = []
    for e in recent:
        href = _md_to_html_href(e.get("path") or "")
        seed = html.escape(e.get("seed") or "?")
        score = float(e.get("dopamine_total") or 0.0)
        activity = e.get("activity") or "research"
        preview = html.escape(_body_preview_text(e.get("path") or ""))
        sections.append(
            f"""<section style="margin-bottom:2.5rem">
  <h2 style="font-family:var(--serif);font-size:1.15rem;margin-bottom:0.5rem">
    <a href="{html.escape(href)}">{seed}</a>
  </h2>
  <div style="margin-bottom:0.75rem">{_dopamine_bar_html(score)} {_activity_chip_html(activity)}</div>
  {f'<p style="color:var(--text-dim);font-size:0.9rem">{preview}</p>' if preview else ''}
</section>"""
        )

    breakdown_block = ""
    if breakdown_rows:
        breakdown_block = f"""
<h2 class="section-title">Activity breakdown</h2>
<table class="stats-table">
  <thead><tr><th>Activity</th><th>Count</th><th>Mean dopamine</th></tr></thead>
  <tbody>{breakdown_rows}</tbody>
</table>
"""

    body = f"""
<p class="meta-line">Today's notebook — {today.isoformat()}. {len(recent)} brief(s) in the last 24 hours.</p>
{breakdown_block}
{''.join(sections) if sections else '<p class="meta-line">Nothing in the last 24 hours.</p>'}
"""
    out = out_dir / "today.html"
    out.write_text(_page("Today", "today", body), encoding="utf-8")
    return out


def write_tree_html(
    nodes: list[dict], journal_dir: Path | str = JOURNAL_DIR
) -> Path:
    """Write journal/tree.html with Mermaid graph + nested HTML tree."""
    out_dir = Path(journal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")

    mermaid_raw = render_tree_mermaid(nodes)
    mermaid_src = re.sub(r"^```mermaid\s*|\s*```$", "", mermaid_raw.strip(), flags=re.MULTILINE)
    mermaid_src = mermaid_src.replace('.md"', '.html"')

    nested = _render_nested_tree(nodes)
    top_rows = ""
    by_score = sorted(
        nodes,
        key=lambda n: (n.get("subtree_score") or n.get("dopamine_total") or 0.0),
        reverse=True,
    )
    for n in by_score[:10]:
        score = n.get("subtree_score") or n.get("dopamine_total") or 0.0
        mut = html.escape(n.get("mutation") or "?")
        d = int(n.get("depth") or 0)
        seed = html.escape((n.get("seed") or "")[:80])
        href = html.escape(_md_to_html_href(n.get("path") or ""))
        top_rows += (
            f"<tr><td class='num'>{float(score):.3f}</td><td>{mut}</td>"
            f"<td class='num'>{d}</td><td><a href='{href}'>{seed}</a></td></tr>\n"
        )

    mermaid_block = ""
    if nodes:
        mermaid_block = f"""
<h2 class="section-title">Graph</h2>
<div class="tree-container">
  <pre class="mermaid">{mermaid_src}</pre>
</div>
"""

    body = f"""
<p class="meta-line">Wander tree — {html.escape(now)}. {len(nodes)} brief(s) in this session.</p>
{mermaid_block}
<h2 class="section-title">Nested view</h2>
{nested}
<h2 class="section-title">Top by subtree score</h2>
<table class="stats-table">
  <thead><tr><th>Score</th><th>Mutation</th><th>Depth</th><th>Seed</th></tr></thead>
  <tbody>{top_rows or '<tr><td colspan="4">No nodes</td></tr>'}</tbody>
</table>
"""
    extra = """
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'dark', securityLevel: 'loose' });
</script>
"""
    out = out_dir / "tree.html"
    out.write_text(_page("Tree", "tree", body, wide=True, extra_head=extra), encoding="utf-8")
    return out


# ----- helpers ----------------------------------------------------------------


def _page(
    title: str,
    nav_active: str,
    body: str,
    *,
    wide: bool = False,
    extra_head: str = "",
) -> str:
    nav_items = [
        ("index", "Index", "index.html"),
        ("today", "Today", "today.html"),
        ("tree", "Tree", "tree.html"),
    ]
    nav_html = "".join(
        f'<a href="{href}" class="{"active" if key == nav_active else ""}">{label}</a>'
        for key, label, href in nav_items
    )
    main_class = "wide" if wide else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · DMN Journal</title>
  <style>{_CSS}</style>
  {extra_head}
</head>
<body>
  <header class="site-header">
    <div class="site-title">default<span>-mode</span>-network</div>
    <nav class="site-nav">{nav_html}</nav>
  </header>
  <main class="{main_class}">
    {body}
  </main>
</body>
</html>
"""


def _activity_chip_html(activity: str) -> str:
    color = ACTIVITY_COLORS.get(activity, "#888")
    return (
        f'<span class="chip" style="background:{color}">'
        f"{html.escape(activity)}</span>"
    )


def _dopamine_bar_html(score: float, max_score: float = 1.0) -> str:
    pct = min(100.0, max(0.0, (float(score) / max(float(max_score), 0.001)) * 100))
    return f"""<div class="dopamine-wrap">
  <span class="dopamine-score">{float(score):.3f}</span>
  <div class="dopamine-bar"><div class="dopamine-fill" style="width:{pct:.1f}%"></div></div>
</div>"""


def _md_to_html_href(md_path: str) -> str:
    if not md_path:
        return "#"
    p = Path(md_path)
    return p.with_suffix(".html").name


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    fm: dict = {}
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") or val.startswith("{"):
            try:
                fm[key] = json.loads(val)
            except json.JSONDecodeError:
                fm[key] = val
        elif val.startswith('"'):
            try:
                fm[key] = json.loads(val)
            except json.JSONDecodeError:
                fm[key] = val.strip('"')
        else:
            fm[key] = val
    return fm, parts[2].lstrip("\n")


def _fix_artifact_path(src: str, html_path: Path) -> str:
    """Return href/src suitable for an HTML file in journal/."""
    if src.startswith(("http://", "https://", "data:", "../")):
        return src
    return artifact_href(html_path, src)


def _md_to_html(body: str, brief_path: Path) -> str:
    """Convert markdown body to HTML, fixing artifact relative paths."""
    text = body.strip()
    html_path = brief_path.with_suffix(".html")

    def _fix_img(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        fixed = _fix_artifact_path(src, html_path)
        return f"![{alt}]({fixed})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _fix_img, text)

    def _fix_audio(m: re.Match) -> str:
        src = m.group(1)
        fixed = _fix_artifact_path(src, html_path)
        return f'<audio controls src="{fixed}"></audio>'

    text = re.sub(
        r'<audio controls src="([^"]+)"></audio>',
        _fix_audio,
        text,
    )
    rendered = _md.render(text)

    def _fix_href(m: re.Match) -> str:
        prefix, url, suffix = m.group(1), m.group(2), m.group(3)
        if url.startswith(("http://", "https://", "#", "data:", "../")):
            return m.group(0)
        fixed = _fix_artifact_path(url, html_path)
        return f'{prefix}{html.escape(fixed)}{suffix}'

    rendered = re.sub(r'(href=")([^"]+)(")', _fix_href, rendered)
    rendered = re.sub(r'(src=")([^"]+)(")', _fix_href, rendered)
    return rendered


def _body_preview_text(path: str, max_lines: int = 4) -> str:
    if not path:
        return ""
    try:
        _, body = _parse_frontmatter(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return ""
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return " ".join(lines[:max_lines])[:400]


def _latest_session_nodes(entries: list[dict]) -> list[dict]:
    """Pick the most recent tree root and return its full subtree."""
    by_id = {int(e["id"]): e for e in entries if e.get("id") is not None}
    if not by_id:
        return []

    roots = [
        e
        for e in by_id.values()
        if e.get("parent_id") is None and e.get("mutation") is not None
    ]
    if not roots:
        return []

    latest_root = max(roots, key=lambda e: e.get("created_at") or 0.0)
    root_id = int(latest_root["id"])

    def in_subtree(n: dict) -> bool:
        cur: Optional[dict] = n
        seen: set[int] = set()
        while cur is not None:
            nid = int(cur["id"])
            if nid == root_id:
                return True
            if nid in seen:
                return False
            seen.add(nid)
            pid = cur.get("parent_id")
            if pid is None:
                return False
            cur = by_id.get(int(pid))
        return False

    return [e for e in by_id.values() if in_subtree(e)]


def _render_nested_tree(nodes: list[dict]) -> str:
    if not nodes:
        return '<p class="meta-line">No tree session recorded yet. Run <code>wander.py</code> for tree mode.</p>'

    by_id = {int(n["id"]): n for n in nodes}
    children_of: dict[int, list[dict]] = {}
    roots: list[dict] = []
    for n in nodes:
        pid = n.get("parent_id")
        if pid is not None and int(pid) in by_id:
            children_of.setdefault(int(pid), []).append(n)
        else:
            roots.append(n)

    def render_node(n: dict) -> str:
        nid = int(n["id"])
        href = html.escape(_md_to_html_href(n.get("path") or ""))
        seed = html.escape((n.get("seed") or "")[:60])
        score = float(n.get("dopamine_total") or 0.0)
        activity = n.get("activity") or "research"
        kids = sorted(
            children_of.get(nid, []),
            key=lambda x: x.get("created_at") or 0.0,
        )
        child_html = "".join(render_node(k) for k in kids)
        nested = f"<ul>{child_html}</ul>" if child_html else ""
        return (
            f'<li class="tree-node">'
            f'<span class="score">{score:.3f}</span> '
            f"{_activity_chip_html(activity)} "
            f'<a href="{href}">{seed}</a>'
            f"{nested}</li>"
        )

    items = "".join(render_node(r) for r in roots)
    return f'<ul class="tree-nested">{items}</ul>'
