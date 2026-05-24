"""CLI to rebuild HTML journal views from existing SQLite + markdown briefs."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import html_journal, store

app = typer.Typer(add_completion=False, help="Rebuild DMN journal HTML views.")
console = Console()


@app.callback()
def _root() -> None:
    """Rebuild DMN journal HTML views."""


@app.command()
def build(
    limit: int = typer.Option(500, "--limit", "-n", help="Max journal rows from SQLite."),
    journal_dir: Path = typer.Option(Path("journal"), "--journal-dir", help="Journal directory."),
) -> None:
    """Rebuild index.html, today.html, tree.html, and per-brief HTML from disk."""
    conn = store.connect()
    try:
        entries = store.list_journal(conn, limit=limit)
        paths = html_journal.build_html_journal(entries, journal_dir=journal_dir)
    finally:
        conn.close()

    console.print("[green]HTML journal rebuilt.[/]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")
    console.print("\n[dim]open journal/index.html[/]")


if __name__ == "__main__":
    app()
