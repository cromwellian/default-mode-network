"""Browser history importer (Chrome, Arc, Brave, Edge, Firefox, Safari).

History databases are read-only-copied to a temp file before opening so the live browser
is never locked. Only title + url + visit_count are read; nothing is uploaded anywhere.

v0.1.3:
  - Default `limit=None` (unbounded). Pass an int to cap.
  - Auto-detect installed browsers via `detected_browsers()`.
  - Each row carries `visit_count` and `weight = log1p(visit_count)` (compresses long tail).
  - Safari SQL fixed: titles live on `history_visits`, not `history_items`.
  - `_clean_url` strips fragments/trailing-slashes/`www.` and drops query strings outside a
    small allow-list (Google `q`, YouTube `v` + `list`, DuckDuckGo `q`).
  - `_clean_title` strips site-name suffixes (" - YouTube", " | LinkedIn", …) before filtering
    so per-page titles look like content rather than fixtures.

v0.1.2: `_clean_history_rows()` drops low-signal entries (homepage / login / generic-domain
title / search-result page / very-short title / per-host floods).
"""
from __future__ import annotations

import math
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dmn.importers._classify import is_allowlisted_host, is_content_url, is_workspace_tool

HOME = Path.home()

PROFILES: dict[str, Optional[Path]] = {
    "chrome": HOME / "Library/Application Support/Google/Chrome/Default/History",
    "arc": HOME / "Library/Application Support/Arc/User Data/Default/History",
    "brave": HOME
    / "Library/Application Support/BraveSoftware/Brave-Browser/Default/History",
    "edge": HOME / "Library/Application Support/Microsoft Edge/Default/History",
    "safari": HOME / "Library/Safari/History.db",
    "firefox": None,  # discovered dynamically below
}

# Hosts whose homepages and bare-root paths are pure noise. Deeper paths on these hosts
# (e.g. github.com/owner/repo) survive, except the allow-list below treats arxiv / wikipedia
# even more permissively.
NOISE_HOSTS: set[str] = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "twitter.com", "www.twitter.com", "x.com", "mobile.twitter.com",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "github.com", "www.github.com",
    "linkedin.com", "www.linkedin.com",
    "gmail.com", "mail.google.com",
    "docs.google.com", "drive.google.com",
    "google.com", "www.google.com",
    "news.ycombinator.com",
    "chatgpt.com", "claude.ai", "gemini.google.com",
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "amazon.com", "www.amazon.com",
    "netflix.com", "www.netflix.com",
}

# Bare-root path forms; any of these on a NOISE_HOST is dropped.
NOISE_ROOT_PATHS: set[str] = {
    "", "/", "/home", "/feed", "/dashboard", "/inbox", "/profile/me",
    "/index.html", "/notifications", "/settings", "/account",
}

# Permissive allow-list — keep entries on these hosts regardless of path/title heuristics.
ALLOW_HOSTS: set[str] = {
    "arxiv.org", "www.arxiv.org",
    "wikipedia.org", "en.wikipedia.org", "fr.wikipedia.org",
    "scholar.google.com",
}

# Generic-title regex: case-insensitive whole-string matches.
_GENERIC_TITLE_RE = re.compile(
    r"^("
    r"new repository|new note|new gist|"
    r"login|log in|sign in|sign up|sign-in|signup|"
    r"home|inbox|dashboard|notifications|settings|account|profile|"
    r"search|search results?|"
    r"google search|.* - google search|.* - youtube|"
    r"youtube|github|google|reddit|twitter|x|linkedin|gmail|drive|"
    r"chatgpt|claude|gemini|"
    r"hacker news|news\.ycombinator\.com|"
    r"netflix|amazon|facebook|instagram|tiktok|whatsapp|"
    r"untitled"
    r")$",
    re.IGNORECASE,
)

# Site-name suffixes to strip from titles. Order matters: longer / more specific first so we
# don't half-strip "YouTube Music" into "Music".
TITLE_SUFFIXES: list[str] = [
    " - YouTube Music",
    " - YouTube",
    " - Google Search",
    " | LinkedIn",
    " · Reddit",
    " : r/",
    " | Reddit",
    " - Reddit",
    " - Wikipedia",
    " | Hacker News",
    " - Hacker News",
    " | Medium",
    " | TechCrunch",
    " | Ars Technica",
    " | The Verge",
    " · GitHub",
    " | GitHub",
    " - GitHub",
    " - Stack Overflow",
    " - StackOverflow",
    " | Stack Overflow",
    " - Twitter",
    " on X",
    " / X",
    " — Mozilla Developer Network",
    " | MDN",
]

# Per-host query-param allow-list. Hosts not in this map drop the entire query string.
KEEP_QUERY_PARAMS: dict[str, set[str]] = {
    "google.com": {"q"},
    "duckduckgo.com": {"q"},
    "youtube.com": {"v", "list"},
}

_TITLE_URL_RE = re.compile(r"^(.*) \((https?://[^)]+)\)$")
_PER_HOST_KEEP = 5


def _webkit_to_unix(webkit_us: int) -> float:
    """Convert WebKit/Chromium timestamp (µs since 1601-01-01) to unix seconds."""
    return (float(webkit_us) / 1_000_000.0) - 11644473600.0


def _safari_to_unix(safari_ts: float) -> float:
    """Convert Safari Core Data timestamp (seconds since 2001-01-01) to unix."""
    return float(safari_ts) + 978307200.0


def _firefox_to_unix(micros: int) -> float:
    """Convert Firefox last_visit_date (µs since unix epoch) to unix seconds."""
    return float(micros) / 1_000_000.0


def _firefox_path() -> Optional[Path]:
    """Locate the most recent Firefox profile's places.sqlite, if any."""
    base = HOME / "Library/Application Support/Firefox/Profiles"
    if not base.exists():
        return None
    for d in sorted(base.iterdir()):
        cand = d / "places.sqlite"
        if cand.exists():
            return cand
    return None


def detected_browsers() -> list[str]:
    """Return the keys of all browsers whose history database is present on this machine."""
    out: list[str] = []
    for key in PROFILES:
        if key == "firefox":
            p = _firefox_path()
        else:
            p = PROFILES.get(key)
        if p and p.exists():
            out.append(key)
    return out


def _clean_title(title: str) -> str:
    """Strip site-name suffixes from a browser-history title."""
    if not title:
        return ""
    out = title.strip()
    lo = out.lower()
    for suf in TITLE_SUFFIXES:
        if lo.endswith(suf.lower()):
            out = out[: -len(suf)]
            break
    return out.rstrip(" -|·—:").strip()


def _clean_url(url: str) -> str:
    """Strip URL noise: fragment, trailing slash, `www.` host prefix, most query params.

    Hosts in `KEEP_QUERY_PARAMS` keep only their listed params (e.g. youtube keeps `v` + `list`,
    google keeps `q`). All other hosts drop the query string entirely.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = ""
    if parts.query and host in KEEP_QUERY_PARAMS:
        keep = KEEP_QUERY_PARAMS[host]
        kept = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if k in keep
        ]
        query = urlencode(kept)
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme or "https", netloc, path, query, ""))


def _normalize_row(row: dict) -> dict:
    """Apply title + URL cleaning to a raw browser row; computes log1p visit-count weight."""
    title = _clean_title(row.get("title") or "")
    url = _clean_url(row.get("url") or "")
    visit_count = max(0, int(row.get("visit_count") or 0))
    weight = float(math.log1p(visit_count)) if visit_count > 0 else 1.0
    out = {
        "title": title,
        "url": url,
        "browser": row.get("browser"),
        "visit_count": visit_count,
        "weight": weight,
    }
    if row.get("last_seen") is not None:
        out["last_seen"] = float(row["last_seen"])
    return out


def find_takeout_chrome(takeout_dir) -> Optional[Path]:
    """Locate Chrome/BrowserHistory.json under a Takeout folder, cheaply.

    Checks the canonical locations before falling back to a bounded recursive scan,
    so huge Takeout trees don't stall the wizard.
    """
    base = Path(takeout_dir).expanduser()
    if not base.exists():
        return None
    for cand in (base / "Chrome" / "BrowserHistory.json",
                 base / "Takeout" / "Chrome" / "BrowserHistory.json"):
        if cand.exists():
            return cand
    for cand in sorted(base.glob("*/Chrome/BrowserHistory.json")):
        return cand
    return None


def import_takeout_history(
    takeout_dir,
    limit: Optional[int] = None,
    keep_noise: bool = False,
    keep_services: bool = False,
) -> list[dict]:
    """Import Google Takeout's Chrome/BrowserHistory.json (synced history).

    Much deeper than the local read: Chrome expires the on-disk database at ~90
    days, while synced history follows the Google account's retention (typically
    18 months or more). Rows are aggregated per URL and run through the same
    noise pipeline as the local importer; source = browser:takeout-chrome.
    """
    import json as _json

    path = find_takeout_chrome(takeout_dir)
    if path is None:
        return []
    size_mb = path.stat().st_size / 2**20
    if size_mb > 1024:
        print(
            f"[dmn] takeout-chrome: {path} is {size_mb:.0f} MB — too large to load "
            "in one piece; skipping. (Streaming support is a known gap.)",
            file=sys.stderr,
        )
        return []
    if size_mb > 100:
        print(
            f"[dmn] takeout-chrome: {path} is {size_mb:.0f} MB — loading may take "
            "a minute and some memory; everything stays on this machine.",
            file=sys.stderr,
        )
    print(
        f"[dmn] takeout-chrome: reading {path} (processed on this machine)",
        file=sys.stderr,
    )
    try:
        payload = _json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"[dmn] takeout-chrome: couldn't parse {path}: {e}", file=sys.stderr)
        return []
    entries = payload.get("Browser History") or []
    by_url: dict[str, dict] = {}
    for e in entries:
        url = e.get("url") or ""
        if not url:
            continue
        usec = e.get("time_usec")
        ts = float(usec) / 1e6 if usec else None
        slot = by_url.setdefault(
            url, {"title": "", "url": url, "visit_count": 0, "last_seen": 0.0}
        )
        slot["visit_count"] += 1
        if e.get("title"):
            slot["title"] = e["title"]
        if ts and ts > slot["last_seen"]:
            slot["last_seen"] = ts
    rows = list(by_url.values())
    print(
        f"[dmn] takeout-chrome: {len(entries):,} visits over {len(by_url):,} pages",
        file=sys.stderr,
    )
    if limit:
        rows.sort(key=lambda r: r["visit_count"], reverse=True)
        rows = rows[:limit]
    normalized = []
    for r in rows:
        n = _normalize_row(r)
        n["browser"] = "takeout-chrome"
        normalized.append(n)
    if keep_noise:
        kept = normalized
        stats_drops: dict = {}
    else:
        kept, stats_drops, _gate = _clean_with_stats(normalized, keep_services=keep_services)
    _report_browser("takeout-chrome", len(rows), len(kept), stats_drops)
    return _to_text_dicts(kept)


def import_history(
    browsers: Iterable[str],
    limit: Optional[int] = None,
    keep_noise: bool = False,
    keep_services: bool = False,
) -> list[dict]:
    """Import title+url+visit_count rows from one or more browsers.

    Returns dicts with `text`, `source`, and `weight`. `limit=None` is unbounded.
    `keep_services=True` (v0.1.4) skips the content-vs-service classifier.
    """
    rows, stats = import_history_with_stats(
        browsers, limit=limit, keep_noise=keep_noise, keep_services=keep_services
    )
    for b, s in stats.items():
        if s.get("available"):
            _report_browser(b, s["raw"], s["kept"], s["drops"])
    return rows


def import_history_with_stats(
    browsers: Iterable[str],
    limit: Optional[int] = None,
    keep_noise: bool = False,
    keep_services: bool = False,
) -> tuple[list[dict], dict]:
    """Like `import_history` but also returns per-browser stats (raw / kept / drops / gate).

    Stats shape: `{"chrome": {"raw": N, "kept": M, "drops": {...}, "service_host_gate": [...],
    "available": True}, ...}`.
    """
    stats: dict[str, dict] = {}
    final: list[dict] = []
    for b in browsers:
        path = _firefox_path() if b == "firefox" else PROFILES.get(b)
        if not path or not path.exists():
            print(
                f"[dmn] {b}: no history database found"
                + (f" at {path}" if path else " (unsupported browser?)"),
                file=sys.stderr,
            )
            stats[b] = {
                "raw": 0,
                "kept": 0,
                "drops": {},
                "service_host_gate": [],
                "available": False,
            }
            continue
        print(
            f"[dmn] {b}: reading local history at {path} "
            "(read-only copy, processed on this machine)",
            file=sys.stderr,
        )
        try:
            raw = _read_raw(path, b, limit)
        except Exception as e:
            print(
                f"[dmn] {b}: couldn't read history (is {b} running? close it and "
                f"retry): {e}",
                file=sys.stderr,
            )
            stats[b] = {
                "raw": 0,
                "kept": 0,
                "drops": {},
                "service_host_gate": [],
                "available": False,
            }
            continue
        normalized = [_normalize_row(r) for r in raw]
        if keep_noise:
            kept, drops, gate = normalized, {}, []
        else:
            kept, drops, gate = _clean_with_stats(
                normalized, keep_services=keep_services
            )
        stats[b] = {
            "raw": len(raw),
            "kept": len(kept),
            "drops": drops,
            "service_host_gate": gate,
            "available": True,
        }
        final.extend(kept)
    return _to_text_dicts(final), stats


def _read_raw(path: Path, browser: str, limit: Optional[int]) -> list[dict]:
    """Copy the SQLite db to a temp file, read it read-only, return raw rows with visit_count."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        shutil.copyfile(path, tmp.name)
        tmp_path = Path(tmp.name)
    try:
        conn = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
        if browser == "safari":
            base_q = (
                "SELECT MAX(history_visits.title) AS title, "
                "history_items.url, "
                "COUNT(history_visits.id) AS visit_count, "
                "MAX(history_visits.visit_time) AS last_visit "
                "FROM history_items "
                "LEFT JOIN history_visits ON history_visits.history_item = history_items.id "
                "GROUP BY history_items.id "
                "ORDER BY last_visit DESC"
            )
        elif browser == "firefox":
            base_q = (
                "SELECT title, url, visit_count, last_visit_date "
                "FROM moz_places ORDER BY last_visit_date DESC"
            )
        else:  # chrome, arc, brave, edge — all use Chromium's `urls` table
            base_q = (
                "SELECT title, url, visit_count, last_visit_time "
                "FROM urls ORDER BY last_visit_time DESC"
            )
        if limit is None:
            rows = conn.execute(base_q).fetchall()
        else:
            rows = conn.execute(base_q + " LIMIT ?", (int(limit),)).fetchall()
        conn.close()
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    out: list[dict] = []
    for row in rows:
        title = (row[0] or "").strip()
        url = (row[1] or "").strip()
        visit_count = int(row[2] or 0) if len(row) > 2 else 0
        last_seen: float | None = None
        if len(row) > 3 and row[3] is not None:
            raw_ts = row[3]
            if browser == "safari":
                last_seen = _safari_to_unix(float(raw_ts))
            elif browser == "firefox":
                last_seen = _firefox_to_unix(int(raw_ts))
            else:
                last_seen = _webkit_to_unix(int(raw_ts))
        entry = {
            "title": title,
            "url": url,
            "browser": browser,
            "visit_count": visit_count,
        }
        if last_seen is not None:
            entry["last_seen"] = last_seen
        out.append(entry)
    return out


def _to_text_dicts(rows: list[dict]) -> list[dict]:
    """Convert rows into final {text, source, weight} dicts.

    Post-cleaning dedup: rows whose `_clean_url` strips down to the same `(title, url)` are
    collapsed into a single interest, summing visit_count and recomputing weight = log1p(sum).
    This matters because (e.g.) `?v=abc&t=30` and `?v=abc&t=120` are the same YouTube video
    and shouldn't appear as separate cluster members.
    """
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        title, url = r.get("title", ""), r.get("url", "")
        text = f"{title} ({url})".strip(" ()")
        if not text:
            continue
        text_clipped = text[:300]
        source = f"browser:{r.get('browser', '?')}"
        key = (text_clipped, source)
        visits = int(r.get("visit_count") or 0)
        if key in by_key:
            entry = by_key[key]
            entry["_visits"] = int(entry.get("_visits", 0)) + visits
            ls = r.get("last_seen")
            if ls is not None:
                entry["_last_seen"] = max(float(entry.get("_last_seen") or 0), float(ls))
        else:
            entry = {
                "text": text_clipped,
                "source": source,
                "_visits": max(visits, 0),
            }
            if r.get("last_seen") is not None:
                entry["_last_seen"] = float(r["last_seen"])
            by_key[key] = entry
            order.append(key)
    out: list[dict] = []
    for key in order:
        entry = by_key[key]
        v = int(entry.pop("_visits", 0))
        entry["weight"] = float(math.log1p(v)) if v > 0 else 1.0
        entry["visit_count"] = v
        if "_last_seen" in entry:
            entry["last_seen"] = entry.pop("_last_seen")
        out.append(entry)
    return out


def _clean_history_rows(rows: list[dict]) -> list[dict]:
    """Pure filter for browser-history rows. Drops low-signal entries (see module docstring).

    Each row may be a raw {title, url, browser, ...} dict or a final {text, source, ...}
    interest dict; the filter parses either form. The `weight` field, if present, is preserved.
    """
    kept, _drops, _gate = _clean_with_stats(rows)
    return kept


# Per-host service-ratio gate: a host with ≥ this many rows AND ≥ this fraction of them
# classified as services is dropped wholesale. Tuned conservatively — allowlisted hosts
# are exempt regardless.
_SERVICE_GATE_MIN_ROWS = 10
_SERVICE_GATE_RATIO = 0.85


def _clean_with_stats(
    rows: list[dict], keep_services: bool = False
) -> tuple[list[dict], dict, list[dict]]:
    """`_clean_history_rows` plus per-category drop counts and per-host service-gate dropoffs.

    Returns `(kept_rows, drops, service_host_gate)` where `service_host_gate` is a list of
    dicts describing hosts dropped wholesale by the service-ratio gate.
    """
    drops = {
        "titleless": 0,
        "generic_title": 0,
        "root_path": 0,
        "short": 0,
        "work_tool": 0,
        "service": 0,
        "service_host_gate": 0,
        "dedup": 0,
    }
    parsed: list[tuple[dict, str, str, str]] = []
    for r in rows:
        title, url = _split_title_url(r)
        host = (urlsplit(url).hostname or "").lower() if url else ""
        if host.startswith("www."):
            host = host[4:]
        parsed.append((r, title, url, host))

    # Pass 1: per-row noise filter (titleless / generic / root / short).
    pre_service: list[tuple[dict, str, str, str]] = []
    for r, title, url, host in parsed:
        title_norm = title.strip()
        title_lower = title_norm.lower()
        if not title_norm or title_lower == url.lower():
            drops["titleless"] += 1
            continue
        in_allow = any(
            host == h or host.endswith("." + h) for h in ALLOW_HOSTS
        )
        is_noise_host = any(
            host == h or host.endswith("." + h) for h in NOISE_HOSTS
        )

        if not in_allow:
            if _GENERIC_TITLE_RE.match(title_lower):
                drops["generic_title"] += 1
                continue
            path = (urlsplit(url).path or "/") if url else "/"
            normalized_path = path.rstrip("/") or "/"
            if is_noise_host and (
                path in NOISE_ROOT_PATHS or normalized_path == "/"
            ):
                drops["root_path"] += 1
                continue
            word_count = len(title_norm.split())
            if word_count < 3 and len(title_norm) < 20:
                drops["short"] += 1
                continue
        pre_service.append((r, title, url, host))

    # Pass 2: content-vs-service classifier.
    filtered: list[tuple[dict, str, str, str]] = []
    per_host_total: Counter = Counter()
    per_host_service: Counter = Counter()
    if keep_services:
        filtered = pre_service
        for r, _t, _u, host in pre_service:
            per_host_total[host or "(unknown)"] += 1
    else:
        for r, title, url, host in pre_service:
            key = host or "(unknown)"
            per_host_total[key] += 1
            if is_workspace_tool(url):
                drops["work_tool"] += 1
                per_host_service[key] += 1
                continue
            ok, _reason = is_content_url(url, title)
            if not ok:
                drops["service"] += 1
                per_host_service[key] += 1
                continue
            filtered.append((r, title, url, host))

    # Pass 3: per-host service-ratio gate. A host with ≥ N rows AND service_ratio > X gets
    # dropped wholesale (allowlisted hosts are exempt — they bypass the classifier anyway).
    gate_drops: list[dict] = []
    if not keep_services:
        gated_hosts: dict[str, dict] = {}
        for host, total in per_host_total.items():
            if total < _SERVICE_GATE_MIN_ROWS:
                continue
            if is_allowlisted_host(host):
                continue
            services = per_host_service.get(host, 0)
            ratio = services / total if total else 0.0
            if ratio > _SERVICE_GATE_RATIO:
                gated_hosts[host] = {
                    "host": host,
                    "total": total,
                    "service": services,
                    "ratio": ratio,
                    "dropped": 0,
                }
        if gated_hosts:
            new_filtered: list[tuple[dict, str, str, str]] = []
            for r, title, url, host in filtered:
                if host in gated_hosts:
                    drops["service_host_gate"] += 1
                    gated_hosts[host]["dropped"] += 1
                    continue
                new_filtered.append((r, title, url, host))
            filtered = new_filtered
            gate_drops = sorted(
                gated_hosts.values(), key=lambda d: -d["service"]
            )

    # Pass 4: per-host dedup (max _PER_HOST_KEEP rows per host).
    per_host: Counter = Counter()
    kept: list[dict] = []
    for r, _title, _url, host in filtered:
        key = host or "(unknown)"
        if per_host[key] >= _PER_HOST_KEEP:
            drops["dedup"] += 1
            continue
        per_host[key] += 1
        kept.append(r)
    return kept, drops, gate_drops


def _split_title_url(row: dict) -> tuple[str, str]:
    """Extract (title, url) from either a raw row or a final {text, source} interest dict."""
    if "title" in row or "url" in row:
        return (row.get("title") or "").strip(), (row.get("url") or "").strip()
    text = row.get("text") or ""
    m = _TITLE_URL_RE.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), ""


def _report_browser(browser: str, raw_total: int, kept_count: int, drops: dict) -> None:
    """Explain the noise filtering in human terms: what was dropped, why, and the dials."""
    try:
        from rich import print as rprint  # type: ignore

        nav = sum(drops.get(k, 0) for k in ("titleless", "generic_title", "root_path", "short"))
        work = drops.get("work_tool", 0)
        service = drops.get("service", 0) + drops.get("service_host_gate", 0)
        dedup = drops.get("dedup", 0)
        rprint(f"  {browser}: {raw_total:,} visits \u2192 [bold]{kept_count:,}[/] kept as taste signal")
        parts = []
        if nav:
            parts.append(f"{nav:,} navigation noise (homepages, search results, short titles)")
        if work:
            parts.append(f"{work:,} work-tool pages (Docs/Sheets/Calendar/Notion/Jira\u2026 \u2014 obligations, not taste)")
        if service:
            parts.append(f"{service:,} service pages (logins, dashboards, checkouts)")
        if dedup:
            parts.append(f"{dedup:,} duplicates")
        if parts:
            rprint(f"[dim]    filtered out: {'; '.join(parts)}[/]")
            rprint("[dim]    keep more: --keep-services (work/service pages) \u00b7 --keep-noise (everything)[/]")
    except Exception:
        pass
