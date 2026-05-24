"""`web_app_sketch` activity (v0.5): self-contained inline HTML/CSS/JS app artifact."""
from __future__ import annotations

import json

from dmn.activities import ActivityContext, ActivityResult, register
from dmn.activities._helpers import extract_fenced, extract_json_meta, slugify, write_artifact_dir
from dmn.generators import Artifact
from dmn.seeds import Seed

SYSTEM = (
    "You design tasteful, self-contained browser apps. Output one complete index.html "
    "file with inline CSS and JavaScript only. No external network, no CDNs, no API keys, "
    "no trackers, no imports. Make it an interactive simulation, tool, or explorable "
    "mini-app related to the seed, with clear controls and understandable code."
)

USER_TEMPLATE = (
    "Seed: {seed_text}\n\n"
    "Create a standalone `index.html` for an interactive simulation/tool/app related to "
    "the seed. Requirements:\n"
    "- all CSS and JavaScript inline\n"
    "- no external network/CDNs/imports\n"
    "- tasteful UI, legible copy, accessible controls\n"
    "- understandable code\n\n"
    "Output two fenced blocks in this order:\n"
    "1. ```html\n<!-- complete index.html -->\n```\n"
    "2. ```json\n{{\"title\": \"...\", \"summary\": \"...\", \"interaction\": \"...\"}}\n```"
)

_DRYRUN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dry-run Signal Mixer</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #f4efe6; }
    main { max-width: 760px; margin: 0 auto; padding: 2rem; }
    .card { border: 1px solid #3b342d; border-radius: 14px; padding: 1.5rem; background: #1c1814; }
    label { display: block; margin: 1rem 0 .3rem; color: #cdbb9b; }
    input { width: 100%; }
    canvas { width: 100%; height: 220px; border: 1px solid #3b342d; border-radius: 10px; }
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Signal Mixer</h1>
      <p>A tiny dry-run app: blend two waves and watch interference patterns shift.</p>
      <label>Frequency A <input id="a" type="range" min="1" max="12" value="3"></label>
      <label>Frequency B <input id="b" type="range" min="1" max="12" value="5"></label>
      <canvas id="plot" width="720" height="220"></canvas>
    </section>
  </main>
  <script>
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    const controls = [...document.querySelectorAll('input')];
    function draw() {
      const a = Number(document.getElementById('a').value);
      const b = Number(document.getElementById('b').value);
      ctx.fillStyle = '#111'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#d6a83a'; ctx.lineWidth = 2; ctx.beginPath();
      for (let x = 0; x < canvas.width; x++) {
        const t = x / canvas.width * Math.PI * 2;
        const y = canvas.height / 2 + (Math.sin(t * a) + Math.sin(t * b)) * 42;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    controls.forEach(input => input.addEventListener('input', draw));
    draw();
  </script>
</body>
</html>"""


class WebAppSketchActivity:
    """Generate a self-contained HTML artifact and link/embed it in the journal."""

    name = "web_app_sketch"
    requires_llm = True
    requires_keys: list[str] = []
    requires_extras: list[str] = []

    def available(self, ctx: ActivityContext) -> bool:
        return True

    def run(self, seed: Seed, ctx: ActivityContext) -> ActivityResult:
        slug = slugify(seed.text, n=50)
        out_dir = write_artifact_dir(ctx.artifact_root, self.name, slug)
        html_text, meta = self._gen(seed, ctx)
        index_path = out_dir / "index.html"
        manifest_path = out_dir / "manifest.json"
        index_path.write_text(html_text.rstrip() + "\n", encoding="utf-8")
        manifest = {
            "activity": self.name,
            "seed": seed.text,
            "title": meta.get("title") or "Web app sketch",
            "summary": meta.get("summary") or "",
            "interaction": meta.get("interaction") or "",
            "entrypoint": "index.html",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        artifact = Artifact(
            modality="html",
            prompt=seed.text,
            bytes_path=index_path,
            url=None,
            mime="text/html",
            generator=self.name,
            meta={"manifest": str(manifest_path), **manifest},
        )
        body = self._render_body(seed, artifact, manifest)
        return ActivityResult(
            title=manifest["title"],
            body_md=body,
            artifacts=[artifact],
            embedding_text=seed.text + " :: " + manifest["title"] + " :: " + manifest["summary"],
            metadata={
                "title": manifest["title"],
                "summary": manifest["summary"],
                "interaction": manifest["interaction"],
                "artifact_paths": [str(index_path), str(manifest_path)],
            },
        )

    def _gen(self, seed: Seed, ctx: ActivityContext) -> tuple[str, dict]:
        if ctx.dry_run:
            return _DRYRUN_HTML, {
                "title": "Dry-run Signal Mixer",
                "summary": "A tiny inline simulation for validating artifact plumbing.",
                "interaction": "Range sliders redraw a blended waveform.",
            }
        try:
            resp = ctx.llm.complete(
                system=SYSTEM,
                user=USER_TEMPLATE.format(seed_text=seed.text),
                max_tokens=4500 if ctx.code_budget in {"medium", "large"} else 3000,
            )
            text = (resp.text or "").strip()
        except Exception:
            text = ""
        html_text = extract_fenced(text, "html") or _DRYRUN_HTML
        meta = extract_json_meta(text) or {
            "title": "Web app sketch",
            "summary": "Self-contained HTML app generated from the seed.",
            "interaction": "Open the artifact and interact with the controls.",
        }
        return html_text, meta

    def _render_body(self, seed: Seed, artifact: Artifact, manifest: dict) -> str:
        return "\n".join(
            [
                f"## Web app sketch: {manifest.get('title') or seed.text}",
                "",
                f"**Seed:** {seed.text}",
                "",
                f"**Summary:** {manifest.get('summary') or '(unspecified)'}",
                "",
                f"**Interaction:** {manifest.get('interaction') or '(open the app)'}",
            ]
        )


register(WebAppSketchActivity())
