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
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dmn.importers._classify import is_allowlisted_host, is_content_url

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
            stats[b] = {
                "raw": 0,
                "kept": 0,
                "drops": {},
                "service_host_gate": [],
                "available": False,
            }
            continue
        try:
            raw = _read_raw(path, b, limit)
        except Exception:
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
    """Print a single per-browser summary line (raw → kept + drop categories)."""
    try:
        from rich import print as rprint  # type: ignore

        rprint(
            f"[dim]  {browser:<8} {raw_total:>6} raw \u2192 {kept_count:>6} kept "
            f"(titleless={drops.get('titleless', 0)}, "
            f"generic={drops.get('generic_title', 0)}, "
            f"root={drops.get('root_path', 0)}, "
            f"short={drops.get('short', 0)}, "
            f"service={drops.get('service', 0)}, "
            f"service_host_gate={drops.get('service_host_gate', 0)}, "
            f"dedup={drops.get('dedup', 0)})[/]"
        )
    except Exception:
        pass
