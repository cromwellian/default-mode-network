"""Taste profile portability — export, import, and (eventually) merge profiles.

Profiles are JSON files (`profile.dmn.json`) carrying interests + cluster centroids + an
embedding-model fingerprint. The vision: swap profiles like mixtapes, build a date-night
shared profile, share anonymized taste vectors with friends. Merge math is a stub in v0.1.1
(it produces a union, not a real blended clustering); see TODO(v0.2) in `dmn/portability.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from dmn import __version__, portability, store

app = typer.Typer(add_completion=False, help="Export, import, and merge DMN taste profiles.")
console = Console()


@app.command()
def export(
    out: Path = typer.Option(
        Path("profile.dmn.json"), "--out", "-o", help="Output JSON path."
    ),
    anonymize: bool = typer.Option(
        False,
        "--anonymize",
        help="Strip raw interest text; keep embeddings + cluster centroids only.",
    ),
) -> None:
    """Serialize the local profile to a portable JSON file."""
    conn = store.connect()
    payload = portability.serialize_profile(conn, anonymize=anonymize)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    n_int = payload["meta"]["interest_count"]
    n_clu = payload["meta"]["cluster_count"]
    tag = " (anonymized)" if anonymize else ""
    console.print(
        f"[green]Exported {n_int} interest(s), {n_clu} cluster(s){tag} -> {out}[/]"
    )


@app.command(name="import")
def import_(
    path: Path = typer.Argument(..., exists=True, help="Path to a profile.dmn.json."),
    replace: bool = typer.Option(False, "--replace", help="Wipe local profile first."),
    append: bool = typer.Option(False, "--append", help="Append into the local profile."),
) -> None:
    """Load a profile JSON into the local SQLite. Choose --replace OR --append."""
    if replace == append:
        console.print("[red]Choose exactly one of --replace or --append.[/]")
        raise typer.Exit(2)
    payload = json.loads(path.read_text())
    conn = store.connect()
    try:
        result = portability.deserialize_profile(
            payload, conn, mode="replace" if replace else "append"
        )
    except ValueError as e:
        console.print(f"[red]Import refused:[/] {e}")
        raise typer.Exit(1)
    console.print(
        f"[green]Imported[/] {result['interests_loaded']} interests, "
        f"{result['clusters_loaded']} clusters from {path}."
    )
    if append:
        console.print(
            "[yellow]Profile marked stale; rebuild clusters with `uv run prepare.py` "
            "or refresh labels with `uv run prepare.py --relabel-only`.[/]"
        )


@app.command()
def merge(
    other: Path = typer.Argument(..., exists=True, help="Path to a second profile.dmn.json."),
    blend: float = typer.Option(
        0.5,
        "--blend",
        help="Blend ratio toward `other` profile (0.0 = local only, 1.0 = other only). "
        "Reserved for v0.2 — currently ignored.",
    ),
    out: Optional[Path] = typer.Option(
        Path("merged.dmn.json"), "--out", "-o", help="Where to write the merged profile."
    ),
) -> None:
    """STUB v0.1.1 — produce a unioned (not blended) profile. Real blend math lands in v0.2."""
    if not 0.0 <= blend <= 1.0:
        console.print("[red]--blend must be between 0 and 1.[/]")
        raise typer.Exit(2)
    console.print(
        "[yellow]merge is a stub in v0.1.1 — emitting a union of both profiles. "
        "Real blended clustering math (using --blend) lands in v0.2.[/]"
    )
    conn = store.connect()
    local_payload = portability.serialize_profile(conn, anonymize=False)
    other_payload = json.loads(other.read_text())
    try:
        merged = portability.union_profiles(local_payload, other_payload)
    except ValueError as e:
        console.print(f"[red]Merge refused:[/] {e}")
        raise typer.Exit(1)
    out_path = out or Path("merged.dmn.json")
    out_path.write_text(json.dumps(merged, indent=2))
    console.print(
        f"[green]Wrote unioned profile[/] ({merged['meta']['interest_count']} interests, "
        f"{merged['meta']['cluster_count']} clusters) -> {out_path}"
    )
    console.print(
        "[yellow]Merged clusters are not coherent until the merged profile is imported "
        "and reclustered with `uv run prepare.py`.[/]"
    )


@app.callback()
def _root() -> None:
    """DMN profile portability — export / import / merge taste profiles."""
    return


if __name__ == "__main__":
    app()
