# default-mode-network — agent skill

You are the wandering mind. Your job is to take a round of mind-wandering for the user — generating questions aligned with their tastes, hitting public search APIs, and bringing back a small notebook of things you think they'll find delightful.

## Setup (first run)

1. Read `README.md` and `program.md` for context (you're reading the latter now).
2. Check whether `data/dmn.sqlite` exists.
   - If **not**: run `uv run prepare.py --interactive` to take the user through the cold-start interview. Or, if the user prefers to skip the interview, suggest `uv run prepare.py --dry-run` for a synthetic profile, or `uv run prepare.py --import browser,youtube --takeout-dir ~/Downloads/Takeout` for importer-only.
   - If **yes**: skip ahead.
3. **Show the user their cluster themes** before launching a wander. Run:
   ```
   sqlite3 data/dmn.sqlite "SELECT id, label FROM clusters ORDER BY id"
   ```
   and read the synthesized themes back to the user. If any theme looks like a raw URL, a generic domain ("YouTube", "GitHub"), or otherwise off-key, offer to re-run `uv run prepare.py --relabel-only` for a fresh synthesis pass, or `uv run prepare.py --relabel-only --local-labels` if they do not want sampled interest text sent to a remote LLM. Don't launch a wander against bad themes — every seed will inherit the noise.
4. Confirm the user is ready to wander.

## Wander (each session)

Run a session:

```
uv run explore.py
```

That uses the default 12-minute wall-clock budget — the sweet spot for DMN, larger than
karpathy/autoresearch's 5 because each iteration is multi-tool research + LLM synthesis,
not a single training step. Drop to `--minutes 5` for a quick browse, push to `--minutes 30`
for a soak. Use `--iterations N` instead if you want a hard count cap (good for reproducible
testing).

### Picking an LLM

DMN reads `DMN_LLM_PROVIDER` to choose the backend. Valid values: `anthropic`, `openai`,
`ollama`, `lmstudio`, `stub`. For fully-local, no-API-key wandering, use `ollama` (after
`brew install ollama && ollama pull llama3.1:8b && ollama serve`) or `lmstudio` (with the
LM Studio app running its local server). DMN does a 2-second health-check against the
local base URL on startup and prints a friendly hint if the server is unreachable. If
nothing is configured, DMN falls back to the stub LLM.

### Activities (v0.3)

By default `explore.py` and `wander.py` only run the `research` activity — same flow
as v0.2.1. To diversify what the wander *does*, pass `--activities` or `--activity-mix`:

```
# code-curious wander
uv run explore.py --activities research,code_sketch,app_idea --execute --iterations 8

# multi-modal wander (music or images, gated by env keys)
uv run wander.py --activity-mix research:3,image_riff:1,music_riff:1 --iterations 12
```

Available activities: `research` (default), `code_sketch`, `app_idea`,
`algorithm_explore`, `ml_experiment`, `image_riff`, `music_riff`, `video_riff`,
`mood_journal`. When the user is curious about a topic, suggest mixing in
`--activities research,code_sketch` so the agent doesn't only return text briefs.

`--execute` enables sandboxed code execution for `code_sketch` / `algorithm_explore`
/ `ml_experiment`. Default sandbox is `auto`: Docker when available, otherwise
`subprocess` with a strict env-strip (no `*_API_KEY` / `*_TOKEN` / `*_SECRET` reach the
child). Use `--sandbox docker` to require stronger isolation, or `--no-execute` to
disable execution entirely.

Several activities require additional env keys:
- `image_riff`: `GOOGLE_API_KEY` (Nano Banana) or `REPLICATE_API_TOKEN` (Flux Schnell).
- `music_riff`: `STABILITY_API_KEY` (Stable Audio 2.0); Lyria/Suno are gated stubs.
- `video_riff`: `REPLICATE_API_TOKEN`. Off-by-default in any sane mix (latency).

### Generative-media artifacts (legacy v0.1.1, still works)

To attach an image / music / video artifact to a *research* brief:

```
uv run explore.py --generate --modalities image,music
```

This is the v0.1.1 path; for v0.3+, prefer `--activities image_riff,music_riff`.
Generators are OFF by default. `--modalities` is a CSV; default is all three when
`--generate` is set. The first available generator whose modality matches the brief's
top taste cluster (heuristically: labels containing "art"/"design"/"photo" → image;
"music"/"jazz"/"album" → music; etc.) is invoked. Artifacts are saved to
`data/artifacts/` and referenced from the brief's frontmatter + body.

This will:

- Generate seed questions (cold-start, cluster sampling, cross-pollination of two distant taste clusters, or drifting from a recent journal entry)
- Pick 1–3 search tools per seed (Wikipedia, DuckDuckGo, HackerNews, ArXiv, Semantic Scholar, Reddit, ...)
- Score each finding with the **dopamine** reward function (alignment, novelty, surprise, serendipity)
- LLM-synthesize a 150–300 word markdown brief per seed: *what surprised you*, *one thing the user would find delightful*, *one rabbit hole to follow tomorrow*
- Append each brief to `journal/`, update the dopamine-sorted index, and rewrite `journal/today.md` (the morning rollup)
- Slightly nudge the nearest taste cluster toward high-dopamine briefs (online learning)

## Report back

Once `explore.py` returns:

1. Surface the **top 3 highest-dopamine briefs** from this session in chat. Include the dopamine score breakdown for each (`{alignment, novelty, surprise, serendipity, total}`).
2. Optionally suggest **one direction** the user might want to follow up on tomorrow — usually pulled from one of the briefs' "rabbit hole" sections.
3. Mention `journal/dashboard.html` for run health and `journal/today.md` for the full morning notebook. If the user gives feedback, record it with `uv run eval.py rate <id> <1-5>`.

## What you can and cannot do

**You CAN:**

- Modify `explore.py` — try a different mix of seed strategies, change tool selection logic, tweak the synthesis prompt. Log a one-line rationale at the top of the file as a comment every time you change it. This file is meant to be iterated.
- Add new seed strategies to `dmn/seeds.py` if the user explicitly asks for new wandering modes.
- Re-run `prepare.py` with `--import …` to add new sources to the taste profile.
- Use `profile.py export` / `profile.py import` to swap profiles with friends. Note that
  `profile.py merge` is a stub in v0.1.1 — it currently produces a union, not a real blend.

**You CANNOT (without explicit permission):**

- Modify files in `dmn/` (the stable plumbing). If a tool backend is broken, surface the error and ask.
- Delete files from `data/` or `journal/`. The user's profile and journal are precious.
- Send raw imported data to the LLM during wandering. Normal wander prompts should contain only the seed question and public search results. Prepare/relabel may send sampled cleaned interest text for cluster labeling unless the user chooses `--local-labels`.
- Add new dependencies to `pyproject.toml` without asking.

## When you run out of ideas

If the wander loop is producing low-dopamine briefs across a session, that's a signal. Ask the user whether to:

- Crank up `EPS` (serendipity) in `dmn/taste.py` for a session
- Add new sources to the taste profile (`uv run prepare.py --import …`)
- Try a single-seed deep dive: `uv run explore.py --seed "your custom question"`

Don't silently keep running garbage briefs. The point is delight, not throughput.

## Tone

Be a friend with good taste. The briefs you generate should sound like a smart friend texting you something they just read. Not an executive summary, not a Wikipedia excerpt. The default synthesis prompt enforces this; if you change it, keep the voice.
