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

from dmn import html_journal, report as report_mod, store
from dmn.llm import get_llm

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


@app.command()
def report(
    run_id: str = typer.Argument(
        None,
        help="Run id to report on. Defaults to the most recent run.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use the stub LLM (offline fallback)."),
) -> None:
    """Generate a NotebookLM-style narrated overview for a wander run."""
    conn = store.connect()
    try:
        if not run_id:
            row = conn.execute(
                "SELECT run_id FROM journal WHERE run_id IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            run_id = row[0] if row else None
        if not run_id:
            console.print("[red]No runs found in the journal.[/]")
            raise typer.Exit(1)
        llm = get_llm(dry_run=dry_run)
        console.print(f"Building report for run [bold]{run_id}[/] (LLM: {llm.name})…")
        md_path = report_mod.build_run_report(conn, run_id, llm=llm)
        if not md_path:
            console.print("[yellow]No briefs for that run; nothing to report.[/]")
            raise typer.Exit(1)
        html_path = html_journal.write_report_html(md_path)
    finally:
        conn.close()
    console.print(f"[green]Report written.[/] {md_path}")
    console.print(f"  open journal/{html_path.name}  (alias: journal/report.html)")


if __name__ == "__main__":
    app()
