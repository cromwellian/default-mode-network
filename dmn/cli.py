"""Console-script wrappers for the root Typer entry points."""
from __future__ import annotations

import typer


def prepare() -> None:
    from prepare import main

    typer.run(main)


def explore() -> None:
    from explore import main

    typer.run(main)


def wander() -> None:
    from wander import main

    typer.run(main)
