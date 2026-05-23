"""`code_sketch` activity (v0.3): tiny self-contained Python program related to the seed.

The LLM produces a ≤100-line, stdlib+numpy-only script with a `__main__` that prints
something useful in ≤30s. We save it to `data/artifacts/code_sketch/<slug>/sketch.py`,
optionally execute it via `dmn.sandbox.run_python`, and render the result as a
markdown brief (code + LLM commentary + execution log).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dmn.activities import ActivityContext, ActivityResult, register
from dmn.activities._helpers import (
    extract_fenced,
    extract_json_meta,
    slugify,
    write_artifact_dir,
)
from dmn.generators import Artifact
from dmn.sandbox import run_python
from dmn.seeds import Seed

SYSTEM = (
    "You write tiny, self-contained Python sketches that explore an idea. "
    "Stdlib + numpy only. No external files, no CLI args, no network. "
    "≤100 lines. Must include `if __name__ == \"__main__\":` and print something "
    "useful in under 30 seconds on a CPU."
)

USER_TEMPLATE = (
    "Seed: {seed_text}\n\n"
    "Write a small Python sketch (≤100 lines, stdlib + numpy only) that explores this. "
    "Be concrete and runnable. Output two fenced blocks in this order:\n\n"
    "1. ```python\n# the program\n```\n"
    "2. ```json\n{{\"what_it_does\": \"...\", \"novelty\": \"...\", \"limitations\": \"...\"}}\n```\n\n"
    "After the blocks, write 2-3 sentences of commentary on what makes this approach "
    "interesting or surprising."
)

_DRYRUN_CODE = '''"""Tiny code sketch (dry-run stub)."""
import math


def main() -> None:
    """Compute the first 10 fibonacci numbers and the golden ratio approx."""
    a, b = 0, 1
    fibs = []
    for _ in range(10):
        fibs.append(a)
        a, b = b, a + b
    ratio = fibs[-1] / fibs[-2] if fibs[-2] else float("nan")
    phi = (1 + math.sqrt(5)) / 2
    print("fibs:", fibs)
    print(f"ratio: {ratio:.6f}, phi: {phi:.6f}, error: {abs(ratio - phi):.2e}")


if __name__ == "__main__":
    main()
'''


class CodeSketchActivity:
    """LLM writes a tiny Python program; optionally we run it in the sandbox."""

    name = "code_sketch"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        # Always callable: dry-run / stub LLM falls back to a canned demo so the pipeline completes.
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        slug = slugify(seed.text, n=50)
        out_dir = write_artifact_dir(ctx.artifact_root, self.name, slug)
        code, meta, commentary = self._gen_code(seed, ctx)
        sketch_path = out_dir / "sketch.py"
        sketch_path.write_text(code + "\n")

        artifacts: list[Artifact] = [
            Artifact(
                modality="text",
                prompt=seed.text,
                bytes_path=sketch_path,
                url=None,
                mime="text/x-python",
                generator=self.name,
                meta={"language": "python"},
            )
        ]

        execution: Optional[dict] = None
        if ctx.execute:
            execution = run_python(sketch_path, mode=ctx.sandbox, timeout=ctx.timeout_s)

        body = self._render_body(seed, code, meta, commentary, execution)
        return ActivityResult(
            title=meta.get("what_it_does") or f"Code sketch: {seed.text[:60]}",
            body_md=body,
            artifacts=artifacts,
            embedding_text=seed.text + " :: " + (meta.get("what_it_does") or "")[:200],
            metadata={
                "what_it_does": meta.get("what_it_does"),
                "novelty": meta.get("novelty"),
                "limitations": meta.get("limitations"),
                "lines": len(code.splitlines()),
            },
            execution=execution,
        )

    def _gen_code(
        self, seed: Seed, ctx: ActivityContext
    ) -> tuple[str, dict, str]:
        """Ask the LLM for code + JSON meta + commentary; fall back to a canned sketch."""
        try:
            resp = ctx.llm.complete(
                system=SYSTEM,
                user=USER_TEMPLATE.format(seed_text=seed.text),
                max_tokens=2000,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        code = extract_fenced(text, "python")
        meta = extract_json_meta(text)
        # Commentary = anything after the last fenced block.
        commentary = ""
        if code:
            tail = text.rsplit("```", 1)[-1].strip()
            commentary = tail if 0 < len(tail) < 500 else ""
        if not code:
            code = _DRYRUN_CODE
            meta = meta or {
                "what_it_does": "fibs and phi (dry-run stub)",
                "novelty": "trivial",
                "limitations": "no LLM was available",
            }
            commentary = commentary or "Stub sketch shipped with the dry-run path."
        return code, meta, commentary

    def _render_body(
        self,
        seed: Seed,
        code: str,
        meta: dict,
        commentary: str,
        execution: Optional[dict],
    ) -> str:
        """Compose the brief markdown for the journal."""
        lines = [
            f"## Code sketch: {meta.get('what_it_does') or seed.text}",
            "",
            f"**Seed:** {seed.text}",
            "",
            "### What it does",
            f"{meta.get('what_it_does') or '(unspecified)'}",
            "",
            "### Code",
            "",
            "```python",
            code,
            "```",
            "",
            "### Notes",
            f"- **Novelty:** {meta.get('novelty') or '(unspecified)'}",
            f"- **Limitations:** {meta.get('limitations') or '(unspecified)'}",
        ]
        if commentary:
            lines.extend(["", "### Commentary", "", commentary])
        if execution is not None:
            lines.extend(self._render_execution(execution))
        return "\n".join(lines)

    def _render_execution(self, execution: dict) -> list[str]:
        """Render the sandbox execution log into the brief."""
        lines = ["", "### Execution"]
        lines.append(
            f"- mode: `{execution.get('mode')}` "
            f"· exit_code: `{execution.get('exit_code')}` "
            f"· duration: `{execution.get('duration_s')}s`"
            + (" · timed out" if execution.get("timed_out") else "")
        )
        stdout = (execution.get("stdout") or "").strip()
        stderr = (execution.get("stderr") or "").strip()
        if stdout:
            lines.extend(["", "**stdout:**", "", "```text", stdout[:2000], "```"])
        if stderr:
            lines.extend(["", "**stderr:**", "", "```text", stderr[:2000], "```"])
        return lines


register(CodeSketchActivity())
