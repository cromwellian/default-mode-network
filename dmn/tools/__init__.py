"""Pluggable research tool registry. Every backend exposes search(query) -> list[ResearchItem]."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class ResearchItem:
    """A normalized research finding shared across all tool backends."""

    title: str
    summary: str
    url: str
    source: str
    raw_text: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


_REGISTRY: dict[str, Callable[..., list[ResearchItem]]] = {}
REGISTRY = _REGISTRY  # public alias for ergonomic imports (mirrors dmn.generators.REGISTRY)


def register(name: str):
    """Decorator: register a search function under the given backend name."""

    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Optional[Callable[..., list[ResearchItem]]]:
    """Return the registered search function, or None."""
    return _REGISTRY.get(name)


def available() -> list[str]:
    """List the names of all currently-registered tool backends."""
    return list(_REGISTRY.keys())


def _register_all() -> None:
    """Auto-discover and register every tool submodule that defines `search`."""
    from importlib import import_module

    for mod in [
        "arxiv",
        "wikipedia",
        "websearch",
        "hackernews",
        "reddit",
        "semantic_scholar",
        "youtube",
        "wikimedia",
        "musicbrainz",
    ]:
        try:
            m = import_module(f"dmn.tools.{mod}")
        except Exception:
            continue
        if hasattr(m, "search"):
            _REGISTRY[mod] = m.search


_register_all()
