"""`app_idea` activity (v0.3): a 1-page markdown PRD inspired by the seed.

Pure-text activity — no execution, no on-disk artifacts beyond the brief itself. Asks
the LLM for a structured PRD with a Mermaid architecture diagram and three risks,
then for a `{name, elevator_pitch, complexity}` JSON meta block.
"""
from __future__ import annotations

from dmn.activities import ActivityContext, ActivityResult, register
from dmn.activities._helpers import extract_json_meta
from dmn.seeds import Seed

SYSTEM = (
    "You sketch tiny apps and tools the user might find delightful to build. "
    "Output is a 1-page markdown PRD: problem, MVP (≤3 features), user, tech stack, "
    "a Mermaid architecture diagram (```mermaid graph LR), 3 risks, 3 unanswered questions."
)

USER_TEMPLATE = (
    "Seed: {seed_text}\n\n"
    "Write the 1-page PRD as described in your system instructions. After the PRD, "
    "append a fenced ```json block with: "
    "`name` (short product name), `elevator_pitch` (≤140 chars), "
    "`complexity` (one of small / medium / large)."
)

_DRYRUN_PRD = """## Problem
A user can't tell which of their dev environments has a stale dependency lock file.

## MVP (3 features)
1. CLI: `lockcheck` — diffs `uv.lock` / `requirements.lock` against the resolver result.
2. Watcher mode: re-runs on `pyproject.toml` save.
3. JSON output for CI.

## User
Solo / small-team Python devs juggling several projects in the same week.

## Tech stack
Python 3.11, click, rich, no DB.

## Architecture
```mermaid
graph LR
  cli[CLI] --> diff[Lock differ]
  diff --> resolver[uv resolver]
  diff --> report[Rich report]
```

## Risks
- Each lock manager has a different format.
- "Stale" is fuzzy when the resolver is non-deterministic.
- Caching could mask the very thing we want to detect.

## Unanswered
- Should we publish a GitHub Action wrapper or stay CLI-only?
- Is there a ≤200-line implementation that's still useful?
- Will users want a fix-it mode, or just a check?

```json
{"name": "lockcheck", "elevator_pitch": "Tell me which of my Python projects has a stale lockfile.", "complexity": "small"}
```
"""


class AppIdeaActivity:
    """LLM writes a 1-page PRD for a small app or tool inspired by the seed."""

    name = "app_idea"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        try:
            resp = ctx.llm.complete(
                system=SYSTEM,
                user=USER_TEMPLATE.format(seed_text=seed.text),
                max_tokens=1500,
            )
            raw = (resp.text or "").strip()
        except Exception:
            raw = ""
        if not raw or "```mermaid" not in raw:
            raw = _DRYRUN_PRD
        meta = extract_json_meta(raw)
        body = "\n".join(
            [
                f"## App idea: {meta.get('name') or seed.text[:60]}",
                "",
                f"**Seed:** {seed.text}",
                "",
                f"**Elevator pitch:** {meta.get('elevator_pitch') or '(none)'}  ",
                f"**Complexity:** {meta.get('complexity') or '(unspecified)'}",
                "",
                raw,
            ]
        )
        return ActivityResult(
            title=meta.get("name") or f"App idea: {seed.text[:60]}",
            body_md=body,
            embedding_text=seed.text + " :: " + (meta.get("elevator_pitch") or "")[:200],
            metadata={
                "name": meta.get("name"),
                "elevator_pitch": meta.get("elevator_pitch"),
                "complexity": meta.get("complexity"),
            },
        )


register(AppIdeaActivity())
