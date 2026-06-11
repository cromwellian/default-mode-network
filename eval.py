"""Rate DMN journal briefs and inspect dopamine-vs-delight alignment."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import typer
from rich.console import Console
from rich.table import Table

from dmn import eval as dmn_eval
from dmn import store

app = typer.Typer(add_completion=False, help="Rate briefs and evaluate dopamine.")
console = Console()


@app.command()
def rate(
    journal_id: int = typer.Argument(..., help="Journal row id to rate."),
    rating: float = typer.Argument(..., help="User delight rating, 1..5."),
    feedback: Optional[str] = typer.Option(None, "--feedback", "-f", help="Optional note."),
) -> None:
    """Attach a 1..5 delight rating to a brief."""
    conn = store.connect()
    row = store.get_journal(conn, journal_id)
    if row is None:
        console.print(f"[red]No journal row with id {journal_id}.[/]")
        raise typer.Exit(1)
    store.update_journal_rating(conn, journal_id, rating, feedback)
    console.print(
        f"[green]Rated[/] #{journal_id} {max(1.0, min(5.0, float(rating))):.1f}/5 — "
        f"{row.get('seed') or Path(row.get('path') or '').stem}"
    )


@app.command()
def report(limit: int = typer.Option(500, "--limit", help="Rated rows to inspect.")) -> None:
    """Show correlations between user ratings and dopamine components."""
    conn = store.connect()
    summary = dmn_eval.summarize(store.list_rated_journal(conn, limit=limit))
    count = int(summary["count"])
    mean = summary["mean_rating"]
    console.print(
        f"[bold]Rated briefs:[/] {count}"
        + (f" · mean rating {float(mean):.2f}/5" if mean is not None else "")
    )
    table = Table(title="Dopamine component correlation with rating")
    table.add_column("component")
    table.add_column("pearson r", justify="right")
    for key, value in (summary.get("correlations") or {}).items():
        table.add_row(key, "n/a" if value is None else f"{float(value):.3f}")
    console.print(table)
    for s in dmn_eval.suggestions(summary):
        console.print(f"  - {s}")


@app.command()
def unrated(limit: int = typer.Option(20, "--limit", help="Rows to show.")) -> None:
    """List recent briefs that do not yet have user ratings."""
    conn = store.connect()
    rows = [e for e in store.list_journal(conn, limit=limit * 3) if e.get("user_rating") is None][:limit]
    table = Table(title="Recent unrated briefs")
    table.add_column("id", justify="right")
    table.add_column("dopamine", justify="right")
    table.add_column("activity")
    table.add_column("seed")
    for e in rows:
        table.add_row(
            str(e.get("id")),
            f"{float(e.get('dopamine_total') or 0.0):.3f}",
            e.get("activity") or "research",
            (e.get("seed") or "")[:80],
        )
    console.print(table)


@app.callback()
def _root() -> None:
    """DMN reward evaluation."""
    return


if __name__ == "__main__":
    app()
