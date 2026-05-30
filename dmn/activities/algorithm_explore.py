"""`algorithm_explore` activity (v0.3): name an algorithm adjacent to the seed,
write a small demo (≤80 lines), include an ASCII or matplotlib visualization,
add a one-paragraph commentary.

Save to `data/artifacts/algorithm_explore/<slug>/`. If `--execute`, run the script
under the chosen sandbox and capture the output (and any `*.png` it writes).
"""
from __future__ import annotations

import re
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
from dmn.grounding import (
    gather_grounding,
    grounding_block,
    grounding_footer,
)
from dmn.fulfillment import compute_activity_fulfillment
from dmn.journal import JOURNAL_DIR
from dmn.paths import artifact_href_for_journal
from dmn.sandbox import run_python
from dmn.seeds import Seed

def _system_for_budget(code_budget: str) -> str:
    budget = (code_budget or "small").lower()
    if budget == "large":
        size = "Up to ~350 lines; multi-function demos are welcome when they clarify the idea."
    elif budget == "medium":
        size = "Up to ~250 lines; multiple functions are welcome."
    else:
        size = "≤80 lines."
    return (
        "You explore algorithms by writing demonstration scripts. "
        f"Constraints: {size} stdlib + numpy. matplotlib is OPTIONAL — if you use it, "
        "write the plot to a file like `plot.png` (do NOT call plt.show()), and ALSO include "
        "a fallback ASCII visualization printed to stdout. Be concrete and educational. "
        "Target NumPy 2.x compatibility: use function forms like `np.ptp(arr)` instead "
        "of removed ndarray methods like `arr.ptp()`."
    )

USER_TEMPLATE = (
    "Seed: {seed_text}\n\n"
    "{grounding}\n\n"
    "1. Pick ONE algorithm or technique adjacent to this seed. Name it clearly.\n"
    "2. Write a runnable demo script that illustrates it ({budget_note}, stdlib + numpy, "
    "matplotlib optional).\n"
    "3. The script's `__main__` should produce both:\n"
    "   - a small ASCII visualization printed to stdout (always)\n"
    "   - a `plot.png` written to the script's working dir (only if matplotlib is "
    "importable; wrap in try/except)\n\n"
    "Output ORDER:\n"
    "- one fenced ```python block with the script\n"
    "- one fenced ```json block: "
    "{{\"algorithm\": \"...\", \"why_interesting\": \"...\", \"complexity\": \"O(...)\"}}\n"
    "- one paragraph of plain-text commentary at the end (no fence)."
)

_DRYRUN_CODE = '''"""Demo: Reservoir sampling (algorithm_explore dry-run)."""
import random


def reservoir(stream, k):
    """Return a uniform-random sample of size `k` from a single-pass stream."""
    out = []
    for i, x in enumerate(stream):
        if i < k:
            out.append(x)
        else:
            j = random.randint(0, i)
            if j < k:
                out[j] = x
    return out


def histogram(samples, n_buckets=20):
    """Tiny ASCII histogram."""
    if not samples:
        return ""
    lo, hi = min(samples), max(samples)
    if hi == lo:
        return f"all={lo}"
    counts = [0] * n_buckets
    for s in samples:
        idx = min(n_buckets - 1, int((s - lo) / (hi - lo) * n_buckets))
        counts[idx] += 1
    return "\\n".join("#" * c for c in counts)


def main() -> None:
    """Sample 100 from a stream of 100k uniform draws; show ASCII hist."""
    random.seed(0)
    stream = (random.gauss(0, 1) for _ in range(100_000))
    sample = reservoir(stream, 100)
    print(f"sampled {len(sample)} items in one pass")
    print(histogram(sample))


if __name__ == "__main__":
    main()
'''


class AlgorithmExploreActivity:
    """Name an adjacent algorithm; write a tiny demo + visualization."""

    name = "algorithm_explore"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = ["matplotlib (optional)"]

    def available(self, ctx: ActivityContext) -> bool:
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        slug = slugify(seed.text, n=50)
        out_dir = write_artifact_dir(ctx.artifact_root, self.name, slug)
        code, meta, commentary = self._gen(seed, ctx)
        sketch_path = out_dir / "demo.py"
        sketch_path.write_text(code + "\n")

        artifacts: list[Artifact] = [
            Artifact(
                modality="text",
                prompt=seed.text,
                bytes_path=sketch_path,
                url=None,
                mime="text/x-python",
                generator=self.name,
                meta={"language": "python", "algorithm": meta.get("algorithm")},
            )
        ]
        execution: Optional[dict] = None
        if ctx.execute:
            execution = run_python(sketch_path, mode=ctx.sandbox, timeout=ctx.timeout_s)
            # Pick up a plot.png if the script wrote one.
            plot_path = out_dir / "plot.png"
            if plot_path.exists():
                artifacts.append(
                    Artifact(
                        modality="image",
                        prompt=meta.get("algorithm") or seed.text,
                        bytes_path=plot_path,
                        url=None,
                        mime="image/png",
                        generator=self.name,
                        meta={"source": "matplotlib"},
                    )
                )

        refs = gather_grounding(seed, ctx)
        body = self._render_body(seed, code, meta, commentary, execution, out_dir) + grounding_footer(refs)
        fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
            activity=self.name,
            body_md=body,
            artifacts=artifacts,
            grounding_items=refs,
            execution=execution,
        )
        return ActivityResult(
            title=meta.get("algorithm") or f"Algorithm: {seed.text[:60]}",
            body_md=body,
            artifacts=artifacts,
            embedding_text=seed.text + " :: " + (meta.get("algorithm") or "") + " :: " + (meta.get("why_interesting") or "")[:300],
            metadata={
                "algorithm": meta.get("algorithm"),
                "why_interesting": meta.get("why_interesting"),
                "complexity": meta.get("complexity"),
                "lines": len(code.splitlines()),
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
                "grounding": [
                    {"title": r.title, "url": r.url, "source": r.source} for r in refs
                ],
            },
            execution=execution,
        )

    def _gen(self, seed: Seed, ctx: ActivityContext) -> tuple[str, dict, str]:
        try:
            max_tokens = 8000 if ctx.code_budget == "medium" else 12000 if ctx.code_budget == "large" else 3000
            budget_note = {
                "medium": "up to about 250 lines",
                "large": "up to about 350 lines",
            }.get(ctx.code_budget, "≤80 lines")
            resp = ctx.llm.complete(
                system=_system_for_budget(ctx.code_budget),
                user=USER_TEMPLATE.format(
                    seed_text=seed.text,
                    budget_note=budget_note,
                    grounding=grounding_block(gather_grounding(seed, ctx)),
                ),
                max_tokens=max_tokens,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        code = extract_fenced(text, "python", salvage=True)
        meta = extract_json_meta(text)
        commentary = ""
        if code:
            tail = text.rsplit("```", 1)[-1].strip()
            commentary = tail if 0 < len(tail) < 800 else ""
        if not code:
            code = _DRYRUN_CODE
            meta = meta or {
                "algorithm": "reservoir sampling",
                "why_interesting": "single-pass uniform sample with O(1) memory per slot",
                "complexity": "O(n)",
            }
            commentary = commentary or "Reservoir sampling is the canonical streaming-uniform-sample trick."
        code = _ensure_numpy2_compat(code)
        return code, meta, commentary

    def _render_body(
        self,
        seed: Seed,
        code: str,
        meta: dict,
        commentary: str,
        execution: Optional[dict],
        out_dir: Path,
    ) -> str:
        plot = out_dir / "plot.png"
        lines = [
            f"## Algorithm: {meta.get('algorithm') or seed.text}",
            "",
            f"**Seed:** {seed.text}",
            "",
            f"- **Why interesting:** {meta.get('why_interesting') or '(unspecified)'}",
            f"- **Complexity:** {meta.get('complexity') or '(unspecified)'}",
            "",
            "### Demo",
            "",
            "```python",
            code,
            "```",
        ]
        if execution is not None:
            lines.extend(self._render_execution(execution))
            if plot.exists():
                plot_href = artifact_href_for_journal(JOURNAL_DIR, plot)
                lines.extend(["", f"![plot]({plot_href})"])
        if commentary:
            lines.extend(["", "### Commentary", "", commentary])
        return "\n".join(lines)

    def _render_execution(self, execution: dict) -> list[str]:
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
            lines.extend(["", "**stderr:**", "", "```text", stderr[:1000], "```"])
        return lines


register(AlgorithmExploreActivity())


def _ensure_numpy2_compat(code: str) -> str:
    """Patch common NumPy 1.x ndarray method calls that fail on NumPy 2.x."""
    code = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.ptp\(\)", r"np.ptp(\1)", code)
    return code
