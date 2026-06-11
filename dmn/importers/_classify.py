"""Content-vs-service URL classifier for the browser importer.

A "service URL" is a transactional interaction with a service: login flows, dashboards,
checkout pages, settings panels, internal corp tools. They pass the v0.1.2 noise filter
(specific paths, non-generic titles) but they aren't *interests* — they're chores.

A "content URL" is something the user reads, watches, or explores: articles, videos,
papers, threads, blog posts, encyclopedia entries.

`is_content_url(url, title)` returns `(bool, reason)`. The reason string is logged in
per-browser drop stats so the user can see *why* each row was filtered.

Tuning principle: false positives (dropping real content) are worse than false negatives
(keeping some services). Rules below match transactional verbs / authentication nouns /
internal-host patterns — NOT topical content. A page *about* loans (article) is fine; a
page that *is* a loan application (`/borrower-app/login`) is not.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit


# 1. ALLOWLIST — high-precision content hosts. Bypass all service rules.
CONTENT_HOST_ALLOWLIST: set[str] = {
    # encyclopedic / academic
    "wikipedia.org", "en.wikipedia.org", "fr.wikipedia.org", "de.wikipedia.org",
    "arxiv.org", "scholar.google.com", "semanticscholar.org",
    # tech aggregators
    "news.ycombinator.com", "lobste.rs",
    # long-form platforms (host = the platform itself)
    "substack.com", "medium.com", "ghost.io",
    # journalism (broad)
    "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com", "bloomberg.com",
    "theatlantic.com", "newyorker.com", "economist.com", "harpers.org",
    "bbc.com", "bbc.co.uk", "news.bbc.co.uk",
    "cnn.com", "reuters.com", "apnews.com", "npr.org",
    "theguardian.com", "latimes.com", "sfgate.com",
    # tech press
    "arstechnica.com", "wired.com", "theverge.com",
    "techcrunch.com", "404media.co", "engadget.com",
    "platformer.news", "stratechery.com",
    # research blogs / forums
    "distill.pub", "lesswrong.com", "alignmentforum.org",
    "deepmind.com", "anthropic.com", "openai.com",
    # specific high-signal blogs the user reads (caught generically by suffix below too)
    "garymarcus.substack.com", "jasonsisney.substack.com", "jabberwocking.com",
    "californiapolicycenter.org",
}

# Suffix matches catch personal blogs / publications hosted on common platforms.
CONTENT_HOST_SUFFIXES: tuple[str, ...] = (
    ".substack.com",
    ".ghost.io",
    ".medium.com",
    ".bearblog.dev",
    ".github.io",
    ".wordpress.com",
    ".blogspot.com",
    ".tumblr.com",
    ".micro.blog",
)

# Specific path prefixes that flip a non-allowlisted host into content (e.g. corporate research blogs).
CONTENT_PATH_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai.com": ("/research", "/blog"),
    "anthropic.com": ("/research", "/news"),
    "deepmind.com": ("/research", "/blog"),
    "google.com": ("/blog",),
    "meta.com": ("/research",),
    "ai.googleblog.com": ("/",),
}

# 2. DROP-WHOLESALE hosts. These have no reasonable content-only path; users can opt-out via --keep-services.
DROP_HOSTS: set[str] = {
    "facebook.com", "m.facebook.com",
    "instagram.com",
    "tiktok.com",
}

# 3. SERVICE INDICATORS — drop if any matches.

# Path segments that indicate transactional flows. We match a segment that starts with one
# of these tokens (so `/borrower-app` matches `borrower`, but `/onboarding` doesn't match `board`).
SERVICE_PATH_SEGMENTS: set[str] = {
    "login", "signin", "sign-in", "signup", "sign-up", "signout", "sign-out", "logout",
    "auth", "authorize", "authentication", "oauth", "openid", "saml", "sso", "idp",
    "account", "accounts", "profile", "preferences", "manage",
    "dashboard", "admin", "console", "control-panel",
    "checkout", "cart", "basket", "order", "orders", "payment", "payments",
    "billing", "invoice", "invoices",
    "inbox", "compose", "drafts", "trash",
    "notifications", "alerts",
    "mfa", "2fa", "verify", "verification", "password", "reset",
    "device", "devices", "sensors", "sensor",
    "appointment", "appointments", "scheduling",
    "prescription", "prescriptions", "refill",
    "medical-records", "lab-results",
    "balance", "transfer", "deposit", "statement", "statements",
    "mortgage", "borrower",
    "upload", "uploads", "download", "downloads",
}

# Same idea but as compiled segment-regex (matches `/borrower-app/...` because of the `(?:[-/]|$)` anchor).
_SERVICE_SEGMENT_RE = re.compile(
    r"(?:^|/)(?:" + "|".join(re.escape(s) for s in SERVICE_PATH_SEGMENTS) + r")(?:[-/]|$)",
    re.IGNORECASE,
)

# Workspace tools: pages with real document titles that pass the noise filter but
# encode obligations, not taste — the single biggest distortion in a knowledge
# worker's history (issue #36: one tester's import was 40% Docs/Sheets/Calendar).
WORKSPACE_TOOL_HOSTS: set[str] = {
    "docs.google.com", "sheets.google.com", "slides.google.com",
    "calendar.google.com", "drive.google.com", "meet.google.com",
    "keep.google.com", "chat.google.com", "mail.google.com",
    "notion.so", "www.notion.so", "linear.app", "airtable.com",
    "asana.com", "app.asana.com", "trello.com",
    "figma.com", "www.figma.com", "miro.com",
    "office.com", "www.office.com", "outlook.office.com", "outlook.live.com",
    "teams.microsoft.com", "teams.live.com",
    "slack.com", "app.slack.com",
    "salesforce.com", "dropbox.com", "www.dropbox.com", "paper.dropbox.com",
}

WORKSPACE_TOOL_SUFFIXES: tuple[str, ...] = (
    ".atlassian.net", ".sharepoint.com", ".slack.com", ".zoom.us",
    ".lightning.force.com", ".monday.com", ".airtable.com",
)


def is_workspace_tool(url: str) -> bool:
    """True when the URL lives on a workspace/productivity tool (work exhaust, not taste)."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except Exception:
        return False
    if host in WORKSPACE_TOOL_HOSTS:
        return True
    return any(host.endswith(suf) for suf in WORKSPACE_TOOL_SUFFIXES)


SERVICE_HOST_PREFIXES: tuple[str, ...] = (
    "login.", "signin.", "signup.", "auth.", "oauth.", "sso.", "idp.", "id.",
    "secure.", "account.", "accounts.", "my.", "myaccount.",
    "portal.", "dashboard.",
    "mail.", "webmail.", "admin.", "console.", "manage.",
    "billing.", "payments.", "checkout.",
    "api.", "gateway.", "proxy.", "static.", "cdn.",
)

SERVICE_HOST_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^myhealth(?:\W|$|\.)"),
    re.compile(r"^my[a-z]*health[a-z]*\."),
    re.compile(r"\.mychart\."),
    re.compile(r"^patient\."),
    re.compile(r"^homeloan"),
    re.compile(r"^myhomeloan"),
    re.compile(r"\.corp\."),
    re.compile(r"\.internal\."),
    re.compile(r"\.prod\.[^.]+\.net$"),
    re.compile(r"^git\.[^.]+\.net$"),
    re.compile(r"^github\.[a-z]+\.net$"),
    re.compile(r"^dagobah\."),  # netflix internal
)

SERVICE_TITLE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^login\b", re.IGNORECASE),
    re.compile(r"^log\s*in\b", re.IGNORECASE),
    re.compile(r"^sign\s*in\b", re.IGNORECASE),
    re.compile(r"^sign\s*up\b", re.IGNORECASE),
    re.compile(r"^sign\s*out\b", re.IGNORECASE),
    re.compile(r"^log\s*out\b", re.IGNORECASE),
    re.compile(r"\bauthentication\b", re.IGNORECASE),
    re.compile(r"^authorize\b", re.IGNORECASE),
    re.compile(r"^authorization\b", re.IGNORECASE),
    re.compile(r"^verify\b", re.IGNORECASE),
    re.compile(r"^two-factor\b", re.IGNORECASE),
    re.compile(r"^2fa\b", re.IGNORECASE),
    re.compile(r"^mfa\b", re.IGNORECASE),
    re.compile(r"^inbox\b", re.IGNORECASE),
    re.compile(r"^dashboard\b", re.IGNORECASE),
    re.compile(r"^settings\b", re.IGNORECASE),
    re.compile(r"^preferences\b", re.IGNORECASE),
    re.compile(r"^my\s+account\b", re.IGNORECASE),
    re.compile(r"^checkout\b", re.IGNORECASE),
    re.compile(r"^cart\b", re.IGNORECASE),
    re.compile(r"\bpayment\s+method\b", re.IGNORECASE),
    re.compile(r"\bbilling\b", re.IGNORECASE),
    re.compile(r"^password\s+reset\b", re.IGNORECASE),
    re.compile(r"^forgot\s+password\b", re.IGNORECASE),
    re.compile(r"^devices?\b", re.IGNORECASE),
    re.compile(r"^cameras?\b", re.IGNORECASE),
    re.compile(r"^sensors?\b", re.IGNORECASE),
    re.compile(r"^loan\s+app\b", re.IGNORECASE),
    re.compile(r"^borrower\s+portal\b", re.IGNORECASE),
    re.compile(r"^my\s*health\b", re.IGNORECASE),
    re.compile(r"^myhealth\b", re.IGNORECASE),
    re.compile(r"\bnetflix\s+authentication\b", re.IGNORECASE),
    re.compile(r"^consumer\s+connect\s+log\s+in\b", re.IGNORECASE),
    re.compile(r"^my\s+health\s+online\b", re.IGNORECASE),
    re.compile(r"^untitled(?:\s+document)?\b", re.IGNORECASE),
)

# Internal/private network indicators.
INTERNAL_HOST_SUFFIXES: tuple[str, ...] = (
    ".local", ".lan", ".internal", ".corp", ".intranet", ".private",
)
_PRIVATE_IP_RE = re.compile(
    r"^(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"
)
_BARE_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _strip_www(host: str) -> str:
    """Strip a leading `www.` from a host."""
    return host[4:] if host.startswith("www.") else host


def is_allowlisted_host(host: str) -> bool:
    """True if `host` (with optional `www.` prefix) is on the content allowlist."""
    h = _strip_www((host or "").lower())
    if not h:
        return False
    if h in CONTENT_HOST_ALLOWLIST:
        return True
    if any(h.endswith(suf) for suf in CONTENT_HOST_SUFFIXES):
        return True
    return False


def _path_conditional_content(host: str, path: str) -> "bool | None":
    """Hosts that have BOTH content and service URLs; return True/False, or None if N/A."""
    if host in {"youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}:
        if path.startswith("/watch") or path.startswith("/shorts/") or path.startswith("/@"):
            return True
        return False
    if host in {"reddit.com", "old.reddit.com", "new.reddit.com"}:
        return "/comments/" in path
    if host in {"twitter.com", "x.com", "mobile.twitter.com"}:
        return bool(re.match(r"^/[^/]+/status/\d+", path))
    if host == "github.com":
        bad_first = {
            "new", "login", "logout", "join", "notifications", "settings",
            "pulls", "issues", "marketplace", "explore", "trending",
            "topics", "collections", "events", "codespaces", "sponsors",
            "account", "billing", "organizations",
        }
        parts = [p for p in path.split("/") if p]
        if not parts:
            return False
        if parts[0] in bad_first:
            return False
        if len(parts) >= 2:
            return True
        return False
    if host in {"linkedin.com", "www.linkedin.com"}:
        if path.startswith("/pulse/"):
            return True
        if re.match(r"^/in/[^/]+/?$", path):
            return True
        return False
    return None


def is_content_url(url: str, title: str = "") -> tuple[bool, str]:
    """Classify a URL+title as content (True) or service (False), with a short reason string.

    Reasons include: `allowlist`, `allowlist-suffix`, `research-path`, `content-path`,
    `default` (all True); `internal-host`, `drop-host`, `service-path-conditional`,
    `service-title`, `service-host-prefix`, `service-host-regex`, `service-path-token` (all False).
    """
    if not url:
        return True, "default"
    try:
        parts = urlsplit(url)
    except Exception:
        return True, "default"
    host = _strip_www((parts.hostname or "").lower())
    path = (parts.path or "/")
    title_norm = (title or "").strip()

    # 1. Allowlist takes precedence over everything else.
    if is_allowlisted_host(host):
        return True, "allowlist"
    for h_prefix, path_prefixes in CONTENT_PATH_PREFIXES.items():
        if host == h_prefix and any(path.startswith(p) for p in path_prefixes):
            return True, "research-path"

    # 2. Internal / private hosts get dropped before any positive rules.
    if host == "localhost" or any(host.endswith(s) for s in INTERNAL_HOST_SUFFIXES):
        return False, "internal-host"
    if _BARE_IP_RE.match(host) or _PRIVATE_IP_RE.match(host):
        return False, "internal-host"

    # 3. Hosts we drop wholesale.
    if host in DROP_HOSTS:
        return False, "drop-host"

    # 4. Path-conditional content (youtube, reddit, twitter, github, linkedin).
    pcc = _path_conditional_content(host, path)
    if pcc is True:
        return True, "content-path"
    if pcc is False:
        return False, "service-path-conditional"

    # 5. Service title patterns (cheap; catches "Login", "Dashboard", "Authentication", etc.).
    if title_norm:
        for pat in SERVICE_TITLE_PATTERNS:
            if pat.search(title_norm):
                return False, "service-title"

    # 6. Service host prefixes (login.*, my.*, accounts.*, …).
    for pref in SERVICE_HOST_PREFIXES:
        if host.startswith(pref):
            return False, "service-host-prefix"

    # 7. Service host regex (myhealth.*, *.mychart., dagobah., *.prod.netflix.net, …).
    for pat in SERVICE_HOST_PATTERNS:
        if pat.search(host):
            return False, "service-host-regex"

    # 8. Service path segments (/login, /checkout, /settings, /borrower-app, …).
    if _SERVICE_SEGMENT_RE.search(path):
        return False, "service-path-token"

    # 9. Default-pass for borderline cases.
    return True, "default"
