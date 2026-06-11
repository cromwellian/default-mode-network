# default-mode-network

*The default mode network is the constellation of brain regions that lights up when you're not doing anything in particular — when you're in the shower, on a walk, drifting before sleep. It's where consciousness wanders, makes weird connections, and quietly does some of its best work. This repo is the same thing, but for an AI: an autonomous wonderer that mind-wanders through your taste profile and brings back things it thinks you'll find delightful. Where [karpathy/autoresearch](https://github.com/karpathy/autoresearch) optimizes a hard objective (val_bpb), DMN optimizes a soft one — virtual dopamine — and is rewarded for serendipity, surprise, and finding rabbit holes you didn't know you wanted to fall into.*

The idea: build a small, local taste profile of *you* (interests, recent obsessions, browser history, papers, songs) and then let an LLM agent wander — generating questions, hitting public search APIs, scoring findings against your taste, and writing short markdown briefs to a journal. You wake up to a notebook of things the AI thought you'd love. Some of them you will. Some of them will be wrong in interesting ways. Both are useful.

## Quick start

In your AI tool (Claude Code, Codex, Cursor), paste:

```text
Clone https://github.com/cromwellian/default-mode-network and set me up — follow its AGENTS.md.
```

Or in a terminal ([the script](scripts/install.sh), 50 lines):

```bash
curl -fsSL https://raw.githubusercontent.com/cromwellian/default-mode-network/main/scripts/install.sh | sh
```

Both end at **`journal/index.html`** — your briefs, sorted by dopamine. Your data stays on your machine; no GPU needed.

<details>
<summary>Manual install · demo without keys</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv, if you don't have it
git clone https://github.com/cromwellian/default-mode-network
cd default-mode-network
uv sync
uv run dmn-setup                # guided: provider, .env, taste profile
uv run explore.py --minutes 5   # first wander
```

No keys, just want to see it move? `uv run prepare.py --dry-run && uv run explore.py --dry-run --iterations 2` runs everything offline with a synthetic profile and a templated LLM.

Optional sharper clusters: `uv sync --extra embeddings` (falls back to fast hash embeddings automatically if it fails on your platform).

</details>

## Choosing your brain: local vs cloud

Your profile, embeddings, journal, and database always stay on your machine. The one thing that can leave is what's sent to the LLM you choose (seed questions, public search results) — so the real choice is who does the thinking:

| option | quality | speed | cost | privacy | needs |
| ------ | ------- | ----- | ---- | ------- | ----- |
| **Claude API key** (recommended) | best | fast | ≈$0.25–0.85 per default wander | prompts go to Anthropic | key from [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI API key** | good | fast | ≈$0.25–0.85 per default wander | prompts go to OpenAI | key from [platform.openai.com](https://platform.openai.com) |
| **Local model** (Ollama / LM Studio) | okay | depends on hardware | free | nothing leaves your machine | Apple-silicon or GPU, ≥8 GB RAM; painful on older CPU-only machines |
| **Demo mode** (no key) | templated stubs | instant | free | fully offline | nothing |

`uv run dmn-setup` walks you through whichever you pick — including installing Ollama and choosing a model your hardware can actually run. Rule of thumb: start with a hosted key for quality; switch to local when privacy matters more than polish (add `--local-labels` and local embeddings for a fully-local pipeline — see [Privacy](#privacy)). Every real (non-demo) run starts by validating your provider with one tiny call and printing an honest cost estimate, so a bad key fails in seconds and demo mode never masquerades as research.

## Teaching DMN your taste

The wizard offers all of these; you can re-run any of them later — prepare **appends** new sources to your profile and re-clusters everything (`--replace` starts fresh).

```bash
uv run prepare.py --interactive                  # 6-question interview, ~2 minutes
uv run prepare.py --import browser --browsers chrome,arc
uv run prepare.py --import youtube,gmail,drive --takeout-dir ~/Downloads/Takeout
```

**What's Google Takeout?** Google's export-your-data service. Go to [takeout.google.com](https://takeout.google.com), click **Deselect all**, then tick only YouTube (watch history), Mail, and/or Drive — a small export arrives in minutes, while select-all takes Google days. Export, unzip, and point `--takeout-dir` at the unzipped folder. (Your Chrome history doesn't need Takeout — `--import browser` reads it straight off your machine.)

Your interests are embedded and clustered into multiple "taste clusters" — a polytopia, not a centroid, so "I like Lisp AND fermentation AND polyrhythms" survives. The seed generator samples one cluster *or* two distant ones (cross-pollination). High-dopamine briefs nudge the nearest cluster as you go, so your profile drifts as you read.

## Wandering

```bash
uv run explore.py                   # flat wander, default 12 minutes
uv run explore.py --minutes 5       # quick browse
uv run explore.py --iterations 12   # reproducible iteration cap

uv run wander.py --iterations 12 --root-count 3 --max-depth 4   # tree-search mode
uv run wander.py --resume           # continue from the open frontier
```

- **`explore.py`** — the flat loop: pick a seed, search, score with the dopamine function, synthesize a brief, log it. This is the agent-modifiable playground file.
- **`wander.py`** — best-first **tree search**: each high-dopamine brief gets expanded via mutation operators (`drift` / `deepen` / `branch` / `retool` / `modality_switch`); a pruner kills weak or duplicate children; dopamine backprops up the ancestor chain. The session tree renders in `journal/tree.html`.

A first Ctrl-C finishes the current iteration and saves everything; a second forces quit. Every brief logs its dopamine breakdown in frontmatter:

`total = α·alignment + β·novelty + γ·surprise + δ·fulfillment` (+ a serendipity bonus when the epsilon-greedy roll fires) — alignment is "stuff you'd like", novelty is "stuff you haven't seen", surprise is "new angles on things you love", fulfillment is "did the tools actually deliver", and serendipity occasionally rewards a low-alignment, high-novelty find to keep your horizons drifting. Constants live in `dmn/taste.py`; see [docs/tuning.md](docs/tuning.md).

## Activities — what does the mind wander on?

Each iteration samples one activity from your `--activities` / `--activity-mix` (default: research only):

| activity            | what it does                                                           | needs                                   |
| ------------------- | ---------------------------------------------------------------------- | --------------------------------------- |
| `research`          | The classic flow: search → score → synthesize a 5-bullet brief.         | (none — works with stub LLM)             |
| `code_sketch`       | LLM writes a ≤100-line stdlib+numpy Python sketch; optionally runs it.  | (none; `--execute` for sandbox run)      |
| `web_app_sketch`    | Self-contained interactive HTML/JS app, embedded in the brief.          | (none)                                   |
| `app_idea`          | 1-page markdown PRD + Mermaid architecture diagram + risks.             | (none)                                   |
| `algorithm_explore` | Picks an adjacent algorithm, writes a tiny demo + ASCII / matplotlib.   | (none; `--extra code` for plots)          |
| `ml_experiment`     | Tiny torch experiment, ≤200 lines, ≤30s on CPU, prints `METRIC: ...`.   | (none to draft; `--extra ml` to run it)   |
| `image_riff`        | LLM rewrites the seed as a visual prompt, calls an image backend.       | `GOOGLE_API_KEY`, `REPLICATE_API_TOKEN`, or an HF token |
| `music_riff`        | LLM rewrites the seed as musical direction, calls an audio backend.     | `STABILITY_API_KEY` (Stable Audio 2.0)   |
| `video_riff`        | Same shape as image; off unless `DMN_ENABLE_VIDEO_RIFFS=1`.             | `RUN_API_KEY` or `REPLICATE_API_TOKEN`   |
| `mood_journal`      | Reflective ~200-word journal entry over your top clusters.              | (none)                                   |

```bash
# Code-curious: research + sketches + app PRDs, with sandboxed execution
uv run wander.py --activities research,code_sketch,app_idea --iterations 12 --execute

# Multimodal: research + image riffs
uv run wander.py --activity-mix research:3,image_riff:1 --iterations 8
```

`--execute` is opt-in and sandboxed (`--sandbox auto|docker|subprocess|none`); subprocess mode strips all `*_API_KEY` / `*_TOKEN` / `*_SECRET` env vars from the child. Media generators are off by default to keep the loop fast and key-free; artifacts land in `data/artifacts/`. Backend tables and every model-override env var: [docs/tuning.md](docs/tuning.md).

## Reading the journal

Each wander writes markdown briefs under `journal/` and regenerates HTML views:
`index.html` (all briefs by dopamine), `dashboard.html` (run health, ratings,
fulfillment), `today.html` (last-24h rollup), `tree.html` (wander session tree), plus one
page per brief. Rebuild anytime with `uv run journal.py build`.

Rate what you read — ratings sharpen the dopamine-vs-you calibration report:

```bash
uv run eval.py unrated            # list unrated briefs with their ids
uv run eval.py rate 42 5 --feedback "exactly the rabbit hole I wanted"
uv run eval.py report
```

## Running as a skill

DMN ships as a one-skill repo: point your coding agent at `program.md` and it knows what to do.

**Claude / Cursor / Codex CLI:**

```
Hi, have a look at program.md and let's go for a wander.
```

The agent checks whether `data/dmn.sqlite` exists, runs setup if not, wanders, then surfaces the top-3 highest-dopamine briefs in chat. A copy of the skill lives at `.claude/skills/default-mode-network/SKILL.md` for native Claude Code discovery.

## How it works

```
prepare.py            — interview + importers + clustering (re-runnable)
explore.py            — the flat wandering loop (the agent edits this)
wander.py             — best-first tree-search wander mode
profiles.py           — export / import / merge taste profiles
journal.py            — rebuild HTML journal / run reports
eval.py               — rate briefs, dopamine-vs-rating report
program.md            — agent instructions
dmn/                  — stable plumbing: taste scoring, seeds, tree, sandbox,
                        embeddings, llm providers, store, journal writers,
                        activities/, tools/, generators/, importers/
data/ , journal/      — local-only profile + output (gitignored)
```

Design choices: small surface area (`explore.py` is the playground, `dmn/` is plumbing), fixed iteration/wall-clock budgets so runs are comparable, SQLite + JSON-blob vectors + a tiny numpy index (no vector DB, no Postgres — you can `cat` your own profile), and graceful degrade everywhere: no key → loud demo mode, no sentence-transformers → hash embeddings, missing tool dep → that tool returns `[]`. The system runs.

Deep dives: [docs/tuning.md](docs/tuning.md) (dopamine constants, wander flags, providers, media backends) · [docs/history.md](docs/history.md) (version notes, migrations) · [docs/modal-api.md](docs/modal-api.md) (remote wanders on Modal).

## Profile portability

Your taste profile is just a SQLite + a small JSON. Export it, swap with friends, merge into a date-night shared profile:

```bash
uv run profiles.py export --out me.dmn.json
uv run profiles.py export --anonymize --out me.anon.dmn.json    # strips raw text
uv run profiles.py import friend.dmn.json --replace             # or --append
uv run profiles.py merge friend.dmn.json --blend 0.5            # see heads-up below
```

**Heads-up:** `merge` produces a unioned profile (concat interests + clusters), marks those clusters as incoherent, and expects you to re-cluster before real use; `--blend` is parsed but currently ignored. Exports carry a per-model embedding fingerprint so you can't accidentally import a profile built in a different embedding space — that would silently corrupt distances. Profiles built by `prepare.py` are also stamped locally with their embedding backend: if it later changes (say, sentence-transformers got pruned from the venv), explore/wander refuse to run and tell you how to fix it.

## Privacy

All profile data — interview answers, browser history, embeddings, journal — lives **locally** in `data/` and `journal/`, both gitignored. None of it is uploaded anywhere.

What can leave your machine is whatever goes to an enabled LLM/search API: during wandering, the seed question and public search results; during prepare/relabel, a small sample of cleaned interest text per cluster for labeling. For a fully-local pipeline: `DMN_LLM_PROVIDER=ollama`, `uv run prepare.py --local-labels`, and local embeddings (`uv sync --extra embeddings`):

```bash
DMN_LLM_PROVIDER=ollama DMN_EMBEDDINGS=st uv run explore.py
```

## Notable forks

(none yet — be the first)

## License

MIT. See `LICENSE`. Have fun. Don't be creepy with someone else's data.
