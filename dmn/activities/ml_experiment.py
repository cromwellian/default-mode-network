"""`ml_experiment` activity (v0.3): a tiny torch + numpy experiment, ≤200 lines, ≤30s on CPU.

Tighter constraints than `code_sketch` / `algorithm_explore`: must use a deterministic
seed, must print a final scalar metric, must complete fast on CPU. If `torch` isn't
installed in the parent env, the activity reports unavailable.
"""
from __future__ import annotations

from importlib.util import find_spec
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
from dmn.sandbox import run_python
from dmn.seeds import Seed

SYSTEM = (
    "You design tiny, runnable ML experiments. Constraints: ≤200 lines; "
    "stdlib + numpy + torch only; deterministic seeds (numpy + torch); "
    "must print a final scalar metric prefixed `METRIC:`; "
    "must complete in ≤30s on CPU. No network. No file I/O outside the script's cwd."
)

USER_TEMPLATE = (
    "Seed: {seed_text}\n\n"
    "{grounding}\n\n"
    "Design a tiny ML experiment that probes a question raised by this seed. "
    "Use synthetic data (numpy). Train for a small number of steps. Print intermediate "
    "loss every few steps and a final `METRIC: <name>=<value>` line.\n\n"
    "Output ORDER:\n"
    "- one fenced ```python block with the script\n"
    "- one fenced ```json block: "
    "{{\"hypothesis\": \"...\", \"metric_name\": \"...\", \"expected_range\": \"low|mid|high\"}}\n"
    "- one paragraph of commentary at the end."
)


class MlExperimentActivity:
    """Run a tiny torch experiment in the sandbox."""

    name = "ml_experiment"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = ["torch"]

    def available(self, ctx: ActivityContext) -> bool:
        # Without torch we can still write the file, but execution would fail. Stay
        # available so the body still renders; mark it cleanly in the metadata.
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        slug = slugify(seed.text, n=50)
        out_dir = write_artifact_dir(ctx.artifact_root, self.name, slug)
        refs = gather_grounding(seed, ctx)
        try:
            resp = ctx.llm.complete(
                system=SYSTEM,
                user=USER_TEMPLATE.format(
                    seed_text=seed.text, grounding=grounding_block(refs)
                ),
                max_tokens=4000,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        code = extract_fenced(text, "python")
        meta = extract_json_meta(text)
        commentary = ""
        if code:
            tail = text.rsplit("```", 1)[-1].strip()
            commentary = tail if 0 < len(tail) < 800 else ""
        if not code:
            body = (
                f"## ML experiment\n\n"
                f"**Seed:** {seed.text}\n\n"
                "_(no LLM available; ml_experiment skipped)_\n"
            )
            fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
                activity=self.name, body_md=body, skipped=True
            )
            return ActivityResult(
                title=f"ML experiment (skipped): {seed.text[:60]}",
                body_md=body,
                embedding_text=seed.text,
                metadata={
                    "skipped": True,
                    "fulfillment": fulfillment,
                    "fulfillment_breakdown": fulfillment_breakdown,
                },
            )
        path = out_dir / "experiment.py"
        path.write_text(code + "\n")
        artifacts: list[Artifact] = [
            Artifact(
                modality="text", prompt=seed.text, bytes_path=path, url=None,
                mime="text/x-python", generator=self.name,
                meta={"language": "python", "hypothesis": meta.get("hypothesis")},
            )
        ]

        execution: Optional[dict] = None
        torch_present = find_spec("torch") is not None
        if ctx.execute:
            if not torch_present:
                execution = {
                    "stdout": "",
                    "stderr": "[ml_experiment] torch not installed in parent env; skipping execution",
                    "exit_code": -10,
                    "duration_s": 0.0,
                    "mode": ctx.sandbox,
                    "timed_out": False,
                }
            else:
                execution = run_python(path, mode=ctx.sandbox, timeout=max(ctx.timeout_s, 60.0))

        body_lines = [
            f"## ML experiment: {meta.get('hypothesis') or seed.text}",
            "",
            f"**Seed:** {seed.text}",
            "",
            f"- **Hypothesis:** {meta.get('hypothesis') or '(unspecified)'}",
            f"- **Metric:** {meta.get('metric_name') or '(unspecified)'}",
            f"- **Expected:** {meta.get('expected_range') or '(unspecified)'}",
            "",
            "### Code",
            "",
            "```python",
            code,
            "```",
        ]
        if execution is not None:
            body_lines.append("")
            body_lines.append("### Execution")
            body_lines.append(
                f"- mode: `{execution.get('mode')}` "
                f"· exit_code: `{execution.get('exit_code')}` "
                f"· duration: `{execution.get('duration_s')}s`"
                + (" · timed out" if execution.get("timed_out") else "")
            )
            stdout = (execution.get("stdout") or "").strip()
            stderr = (execution.get("stderr") or "").strip()
            if stdout:
                body_lines.extend(["", "**stdout:**", "", "```text", stdout[:2500], "```"])
            if stderr:
                body_lines.extend(["", "**stderr:**", "", "```text", stderr[:1000], "```"])
        if commentary:
            body_lines.extend(["", "### Commentary", "", commentary])

        body = "\n".join(body_lines) + grounding_footer(refs)
        fulfillment, fulfillment_breakdown = compute_activity_fulfillment(
            activity=self.name,
            body_md=body,
            artifacts=artifacts,
            grounding_items=refs,
            execution=execution,
        )
        return ActivityResult(
            title=meta.get("hypothesis") or f"ML experiment: {seed.text[:60]}",
            body_md=body,
            artifacts=artifacts,
            embedding_text=seed.text + " :: " + (meta.get("hypothesis") or "")[:300],
            metadata={
                "hypothesis": meta.get("hypothesis"),
                "metric_name": meta.get("metric_name"),
                "expected_range": meta.get("expected_range"),
                "torch_present": torch_present,
                "lines": len(code.splitlines()),
                "fulfillment": fulfillment,
                "fulfillment_breakdown": fulfillment_breakdown,
                "grounding": [
                    {"title": r.title, "url": r.url, "source": r.source} for r in refs
                ],
            },
            execution=execution,
        )


register(MlExperimentActivity())
