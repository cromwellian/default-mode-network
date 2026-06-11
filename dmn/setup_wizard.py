"""First-run setup wizard: pick how DMN thinks, get it working, build a taste profile.

Run from the repo root:

    uv run dmn-setup

The wizard guides provider choice (hosted Claude key / local model via Ollama / demo
mode), validates the choice with one cheap call, writes `.env`, then drives `prepare.py`
to build the taste profile, and ends by printing the exact next command. The local path
is hardware-aware: it recommends a model size the machine can actually run, walks through
installing Ollama and pulling the model, and health-checks the server.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from getpass import getpass
from pathlib import Path

from rich.console import Console

from dmn import llm as llm_mod

console = Console()

ENV_PATH = Path(".env")
OLLAMA_URL = llm_mod._LOCAL_DEFAULTS["ollama"]["base_url"]

# Rough download sizes so users aren't surprised mid-pull.
_MODEL_SIZES_GB = {"llama3.1:8b": 4.9, "llama3.2:3b": 2.0}


def _machine_arch() -> str:
    """platform.machine(), corrected for Rosetta.

    An x86_64 Python toolchain on Apple silicon reports x86_64 and would steer a
    perfectly capable machine away from local models; sysctl sees through it.
    """
    arch = platform.machine()
    if sys.platform == "darwin" and arch.lower() == "x86_64":
        try:
            probe = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            lines = probe.splitlines()
            # Two independent signals: the arm64 oid, or a brand line like
            # "Apple M2 Max" (covers the oid query failing). Prefix-anchored so
            # no Intel brand string can ever match.
            if lines[:1] == ["1"] or any(l.startswith("Apple ") for l in lines):
                return "arm64"
        except Exception:
            pass
    return arch


def _total_ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            return int(out.stdout.strip()) / 2**30
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return None


def recommend_local_model(arch: str, ram_gb: float | None) -> tuple[str | None, str]:
    """Pick a local model this machine can actually run. (None, note) = don't go local."""
    arm = arch.lower() in ("arm64", "aarch64")
    if ram_gb is None:
        if arm:
            return (
                "llama3.2:3b",
                "Couldn't detect RAM — defaulting to a small 3B model. If you have "
                "≥16 GB, pick llama3.1:8b instead.",
            )
        return (
            None,
            "Couldn't detect RAM, and CPU-only x86 is slow for local models — a hosted "
            "key or demo mode is the safer bet.",
        )
    ram = ram_gb
    if not arm:
        if ram >= 16:
            return (
                "llama3.2:3b",
                "CPU-only x86: an 8B model would crawl here. A 3B model is usable for "
                "tinkering, but a hosted API will be far better.",
            )
        return (
            None,
            "This machine (CPU-only x86, <16 GB RAM) is too slow for a satisfying "
            "local model — a hosted key or demo mode will serve you better.",
        )
    if ram >= 16:
        return "llama3.1:8b", "Apple-silicon/ARM with ≥16 GB RAM — an 8B model runs well."
    if ram >= 8:
        return "llama3.2:3b", "8–16 GB RAM — a 3B model fits comfortably; an 8B would swap."
    return (
        None,
        "Under 8 GB RAM — local models will struggle; a hosted key or demo mode is better.",
    )


def update_env_file(updates: dict[str, str | None], path: Path = ENV_PATH) -> None:
    """Set keys in .env in place, preserving unrelated lines and comments.

    A value of None removes the key — used when switching providers so a stale
    DMN_LLM_PROVIDER can't silently override the new choice on the next run.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = None
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            value = remaining.pop(key)
            if value is not None:
                out.append(f"{key}={value}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in remaining.items() if v is not None)
    path.write_text("\n".join(out) + "\n")
    path.chmod(0o600)  # .env holds API keys


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    ans = input(f"{prompt}{suffix} ").strip()
    return ans or default


def _confirm(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {hint} ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def _setup_anthropic() -> bool:
    console.print(
        "\nGet a key at [bold]https://console.anthropic.com[/] → API keys. "
        "A wander costs roughly $0.25–0.85."
    )
    for _ in range(3):
        key = getpass("Paste your ANTHROPIC_API_KEY (input hidden): ").strip()
        if not key:
            console.print("[yellow]Empty — let's try again (Ctrl-C to abort).[/]")
            continue
        os.environ["ANTHROPIC_API_KEY"] = key
        os.environ.pop("DMN_LLM_PROVIDER", None)
        console.print("Checking the key with one tiny API call…")
        err = llm_mod.preflight(llm_mod.get_llm("anthropic"))
        if err is None:
            update_env_file(
                {"ANTHROPIC_API_KEY": key, "DMN_LLM_PROVIDER": None, "DMN_LLM_MODEL": None}
            )
            console.print("[green]Key works. Saved to .env (gitignored).[/]")
            return True
        console.print(f"[red]{err}[/]")
    console.print("[yellow]Couldn't validate a key after 3 tries.[/]")
    return False


def _wait_for_ollama(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if llm_mod._health_check(OLLAMA_URL):
            return True
        time.sleep(1.0)
    return False


def _setup_ollama(model: str) -> bool:
    if not shutil.which("ollama"):
        console.print("\nOllama isn't installed yet.")
        if shutil.which("brew") and _confirm("Install it now with Homebrew?"):
            if subprocess.run(["brew", "install", "ollama"]).returncode != 0:
                console.print("[red]brew install ollama failed — see output above.[/]")
                return False
        else:
            console.print(
                "Install it from [bold]https://ollama.com/download[/], "
                "then run [bold]uv run dmn-setup[/] again."
            )
            return False
    if not llm_mod._health_check(OLLAMA_URL):
        started = False
        if shutil.which("brew"):
            console.print("Starting the Ollama server (brew services start ollama)…")
            subprocess.run(["brew", "services", "start", "ollama"], capture_output=True)
            started = _wait_for_ollama()
        if not started:
            console.print(
                "[yellow]Ollama server isn't responding. In another terminal, run "
                "[bold]ollama serve[/], then press Enter here.[/]"
            )
            input()
            if not _wait_for_ollama(10):
                console.print(f"[red]Still can't reach {OLLAMA_URL} — aborting local setup.[/]")
                return False
    size = _MODEL_SIZES_GB.get(model)
    size_note = f" (~{size:g} GB download)" if size else ""
    if not _confirm(f"Pull the model [bold]{model}[/]{size_note}?"):
        return False
    if subprocess.run(["ollama", "pull", model]).returncode != 0:
        console.print("[red]ollama pull failed — see output above.[/]")
        return False
    os.environ["DMN_LLM_PROVIDER"] = "ollama"
    os.environ["DMN_LLM_MODEL"] = model
    console.print("Checking the model with one tiny completion…")
    err = llm_mod.preflight(llm_mod.get_llm("ollama"), timeout_s=120)
    if err:
        console.print(f"[red]{err}[/]")
        return False
    update_env_file({"DMN_LLM_PROVIDER": "ollama", "DMN_LLM_MODEL": model})
    console.print("[green]Local model works. Saved provider settings to .env.[/]")
    return True


def _profile_exists() -> bool:
    db = Path("data/dmn.sqlite")
    if not db.exists():
        return False
    try:
        from dmn import store

        conn = store.connect(db)
        n = len(store.list_interests(conn))
        conn.close()
        return n > 0
    except Exception:
        return False


def _ask_path(prompt: str, ask=_ask) -> str | None:
    """Ask for a path until it exists; empty answer skips."""
    while True:
        raw = ask(prompt + " (or Enter to skip)")
        if not raw:
            return None
        p = Path(raw).expanduser()
        if p.exists():
            return str(p)
        console.print(f"[yellow]Can't find {p} — check the path.[/]")


def source_walk(ask=_ask, confirm=_confirm, say=None) -> list[str]:
    """Walk taste sources one at a time (yes → details, skip → next); returns prepare args.

    Everything lands in ONE prepare run because runs replace the profile (#33).
    """
    say = say or console.print
    args: list[str] = []
    imports: list[str] = []

    say("\nLet's gather your taste — one source at a time. Skip anything freely.")

    if confirm("1/5 Interview — 6 quick questions about what you love right now?"):
        args.append("--interactive")

    say(
        "[dim]Browser history: DMN reads your browser's local database (Chrome/Arc/"
        "Brave/Edge/Firefox/Safari) — a read-only copy, processed on this machine. "
        "Holds roughly the last 90 days. Close the browser first. Noise (homepages, "
        "search results, per-site floods) is filtered out.[/]"
    )
    if confirm("2/5 Import browser history?"):
        imports.append("browser")

    say(
        "[dim]YouTube watch history: the single richest 'what I consume' signal. "
        "Comes from a Google Takeout export (takeout.google.com → Deselect all → "
        "tick YouTube → minutes, not days). Gmail/Drive ride the same folder.[/]"
    )
    if confirm("3/5 Import YouTube history from a Takeout folder?"):
        path = _ask_path("Path to the unzipped Takeout folder", ask=ask)
        if path:
            imports.append("youtube")
            args += ["--takeout-dir", path]
            if confirm("Also import Gmail-sent and Drive titles from it? (weaker taste signal)", default=False):
                imports += ["gmail", "drive"]

    if confirm("4/5 Import Twitter/X likes from an archive export?", default=False):
        path = _ask_path("Path to the unzipped Twitter archive folder", ask=ask)
        if path:
            imports.append("twitter")
            args += ["--twitter-dir", path]

    say(
        "[dim]Spotify, Goodreads, Readwise, saved-links apps — any CSV with a "
        "Title/title/Highlight/text column works. (Spotify: exportify.net exports "
        "Liked Songs to CSV in minutes.)[/]"
    )
    if confirm("5/5 Import a CSV export like that?", default=False):
        path = _ask_path("Path to the CSV file", ask=ask)
        if path:
            imports.append("readwise")
            args += ["--readwise-csv", path]

    extra = ask(
        "Any other data source you wish to include? Supported today: the ones above; "
        "anything else exports to CSV (title/text column) and comes in via 5. "
        "Name it and I'll note it, or Enter to finish:"
    )
    if extra:
        say(
            f"[yellow]Noted: '{extra}' isn't natively supported yet — if it exports "
            "to CSV, re-run dmn-setup and feed it via the CSV step. Consider filing "
            "an importer request on GitHub.[/]"
        )

    if imports:
        args += ["--import", ",".join(imports)]
    return args


def _build_profile(demo_only: bool) -> int:
    rebuild = False
    if _profile_exists():
        console.print("\nYou already have a taste profile in [bold]data/dmn.sqlite[/].")
        if not _confirm("Rebuild it from scratch? (No keeps the existing one)", default=False):
            return 0
        rebuild = True

    cmd = [sys.executable, "prepare.py"]
    cmd += source_walk()
    if demo_only:
        cmd.append("--dry-run")
    if len(cmd) == 2:
        console.print("[yellow]No sources chosen — seeding a synthetic demo profile.[/]")
        cmd.append("--dry-run")
    if rebuild:
        cmd.append("--replace")  # already confirmed above; avoid a double prompt
    console.print(f"[dim]Running: {' '.join(cmd[1:])}[/]\n")
    return subprocess.run(cmd).returncode


def run() -> None:
    """Entry point for the `dmn-setup` console script."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    try:
        _run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup aborted — re-run `uv run dmn-setup` any time.[/]")
        raise SystemExit(130)
    except EOFError:
        console.print("\n[red]Input ended — run dmn-setup in an interactive terminal.[/]")
        raise SystemExit(1)


def _run() -> None:
    console.print("[bold magenta]DMN setup[/] — let's get you wandering.\n")
    if not Path("prepare.py").exists():
        console.print("[red]Run this from the default-mode-network repo root.[/]")
        raise SystemExit(1)
    if not sys.stdin.isatty():
        console.print("[red]dmn-setup is interactive — run it in a terminal.[/]")
        raise SystemExit(1)

    arch, ram = _machine_arch(), _total_ram_gb()
    ram_note = f"{ram:.0f} GB RAM" if ram else "unknown RAM"
    rosetta = " (Apple silicon under a Rosetta toolchain)" if arch != platform.machine() else ""
    console.print(f"Machine: [bold]{arch}[/]{rosetta}, {ram_note}")
    local_model, local_note = recommend_local_model(arch, ram)

    console.print("\nHow should DMN think?")
    console.print(
        "  1) [bold]Claude API key[/] — best quality, ≈$0.25–0.85 per wander (recommended)"
    )
    local_suffix = f" — would use [bold]{local_model}[/]" if local_model else ""
    console.print(f"  2) Local model via Ollama — free and private{local_suffix}")
    console.print(f"     [dim]{local_note}[/]")
    console.print("  3) Demo mode — no key; templated briefs, just to see the plumbing")
    choice = _ask("Choose 1, 2 or 3:", default="1")

    provider_ok = False
    if choice == "2":
        if local_model is None:
            if not _confirm("Local really isn't a good fit here. Try anyway with a 3B model?", default=False):
                raise SystemExit(1)
            local_model = "llama3.2:3b"
        provider_ok = _setup_ollama(local_model)
    elif choice == "3":
        console.print("[yellow]Demo mode: briefs will be templated stubs, not real research.[/]")
        provider_ok = True
    else:
        provider_ok = _setup_anthropic()

    if not provider_ok:
        console.print(
            "\n[yellow]Provider setup didn't finish. You can re-run `uv run dmn-setup` "
            "any time — your answers aren't lost.[/]"
        )
        raise SystemExit(1)

    rc = _build_profile(demo_only=(choice == "3"))
    if rc != 0:
        console.print(f"[red]prepare.py exited with {rc} — fix the issue above and re-run.[/]")
        raise SystemExit(rc)

    console.print("\n[bold green]Setup complete.[/] Get familiar with a small taster first:")
    if choice == "3":
        console.print("  [bold]uv run explore.py --dry-run --iterations 2[/]   (demo output)")
    else:
        console.print("  [bold]uv run explore.py --iterations 3[/]   (~2–3 min, pennies)")
        console.print(
            "Then a full session — default 12 minutes; pick what it produces:\n"
            "  [bold]uv run explore.py --minutes 12 --activities research,code_sketch[/]\n"
            "  (menu: research, code_sketch, web_app_sketch, app_idea, algorithm_explore,\n"
            "   mood_journal; image/music/video riffs need extra keys — docs/tuning.md)"
        )
    console.print("Then open [bold]journal/index.html[/] in a browser to read what it found.")
