# default-mode-network

*The default mode network is the constellation of brain regions that lights up when you're not doing anything in particular — when you're in the shower, on a walk, drifting before sleep. It's where consciousness wanders, makes weird connections, and quietly does some of its best work. This repo is the same thing, but for an AI: an autonomous wonderer that mind-wanders through your taste profile and brings back things it thinks you'll find delightful. Where [karpathy/autoresearch](https://github.com/karpathy/autoresearch) optimizes a hard objective (val_bpb), DMN optimizes a soft one — virtual dopamine — and is rewarded for serendipity, surprise, and finding rabbit holes you didn't know you wanted to fall into.*

The idea: build a small, local taste profile of *you* (interests, recent obsessions, browser history, papers, songs) and then let an LLM agent wander — generating questions, hitting public search APIs, scoring findings against your taste, and writing short markdown briefs to a journal. You wake up to a notebook of things the AI thought you'd love. Some of them you will. Some of them will be wrong in interesting ways. Both are useful.

There is no `val_bpb` here. There is no "right answer". The system rewards *alignment with your tastes*, *novelty against what you've already read*, and *surprise* — angles on familiar topics it hasn't surfaced before. With a small chance of pure serendipity, just in case your taste profile has converged a little too neatly.

## How labels are made

DMN's seed quality lives or dies on its cluster labels. Done badly, the seed generator ends up asking "What's a fresh angle on YouTube (https://www.youtube.com/?" — which is what happened in v0.1.0 when labels were just the nearest-centroid interest's raw text. v0.1.2 fixes it in two places:

1. **Browser-importer cleanup** (`dmn/importers/browser.py`). Before any clustering, the importer drops homepage visits, login pages, generic-domain titles ("YouTube", "GitHub", "Gmail", …), search-result pages, very-short titles, and per-host floods (max 5 entries per host). Pass `--keep-noise` to bypass.
2. **Synthesized cluster labels** (`dmn/labeling.py`). After clustering in embedding space, DMN picks the 12 nearest-medoid members per cluster and asks the configured LLM for `{"theme": "...", "subtopics": [...], "rationale": "..."}` — a *thematic* label, not a verbatim title. The theme is stored as the cluster's `label` column (back-compat); the subtopics and rationale go into a new `meta` JSON column. Subtopics flow through to seed prompts, so questions like "What's a fresh angle on contrarian nutrition science?" replace "What's a fresh angle on Oreos vs Statins (https://...)". Use `--local-labels` to force the deterministic local labeler so sampled interest text is not sent to a remote LLM during prepare/relabel.

If a cluster's theme looks off, run `uv run prepare.py --relabel-only` to re-synthesize labels against the existing profile (no re-import, no re-cluster — fast). The dry-run path (`--dry-run`) uses a deterministic word-frequency fallback so you can iterate without burning API tokens.

## Clustering (v0.4)

DMN defaults to **HDBSCAN** on L2-normalized embeddings with **medoid** cluster representatives (not K-means centroids). Sample weights combine `log1p(visit_count)` with exponential recency decay; manual and interview interests are pinned as high-recency seeds.

```bash
# Default: HDBSCAN + recency-weighted taste profile
uv run prepare.py --import browser

# Explicit method selection
uv run prepare.py --cluster-method hdbscan   # default
uv run prepare.py --cluster-method kmeans    # legacy K-means (k=8)
uv run prepare.py --cluster-method gmm       # Gaussian mixture + BIC k-selection

# Tune recency half-life (days) and HDBSCAN min cluster size
uv run prepare.py --recency-half-life 90 --min-cluster-size 20
uv run prepare.py --local-labels  # privacy mode for cluster labels
```

**Noise bucket:** points HDBSCAN cannot assign land in an `ambient / unclustered` cluster (`is_noise=1`). Seed sampling skips this bucket unless a small serendipity roll hits (~5%). Upgrade an existing profile:

```bash
uv run scripts/migrate_v0_4.py
```

Inspect clusters (medoid text, method, noise flag, silhouette):

```bash
uv run scripts/inspect_clusters.py --top 10
```

## Wander controls and web apps (v0.5)

Tree-mode wander now records a `run_id` for each invocation, stores per-node visit /
expansion metrics in SQLite, and keeps `journal/tree.html` focused on the latest run when
session data is available. You can bound the live frontier with beam pruning and stop when
new expansions stop improving dopamine:

```bash
uv run wander.py \
  --root-count 8 \
  --max-depth 5 \
  --beam-width 4 \
  --patience 6 \
  --min-improvement 0.01 \
  --activity-mix 'research:4,code_sketch:2,web_app_sketch:2,app_idea:1,algorithm_explore:1' \
  --code-budget medium \
  --execute
```

Cross-pollination now samples cluster pairs from a moderate distance band instead of always
forcing the farthest pair, which tends to produce seeds with more useful overlap. Image and
music riffs are strongly downweighted unless the seed or cluster text suggests visual/audio
form would help explain the concept, or you explicitly request only that activity.

`web_app_sketch` is a new opt-in activity that writes a self-contained
`data/artifacts/web_app_sketch/<slug>/index.html` plus `manifest.json`. Generated journal
briefs link to the app and embed it in a sandboxed iframe. Code activities accept
`--code-budget small|medium|large`; `medium` allows larger multi-function sketches and a
longer sandbox timeout, while `large` allows larger simple artifacts up to a 120s timeout.

## Upgrading from earlier dev versions

If you ran v0.1.0 / v0.1.1 against your real browser history, your existing SQLite carries the noisy rows and URL-flavored cluster labels. One-shot fix:

```bash
uv run scripts/migrate_v0_1_2.py
```

It re-runs the new browser filter against existing `browser:*` interests (deletes the noisy ones), re-clusters, and re-labels via the LLM. Idempotent — running it again is a no-op.

## How it works

The repo is deliberately small and only a few files matter:

- **`prepare.py`** — one-time setup. Runs an interview, optionally imports browser history / Google Takeout / Twitter likes, embeds the interest items, k-means clusters them into "taste clusters", and writes everything to `data/dmn.sqlite`. Edit if you want; doesn't change much per run.
- **`explore.py`** — the single agent-modifiable file, analogous to karpathy's `train.py`. The wandering loop: pick a seed, plan tools, search, score with the dopamine function, synthesize a brief, log it. **The agent is encouraged to iterate on this file** — try different seed-picking strategies, different ratios of cluster vs. cross-pollination, different synthesis prompts, etc.
- **`wander.py`** *(v0.2)* — a sibling entry point that runs best-first **tree-search** instead of flat exploration. Each high-dopamine brief gets expanded via mutation operators (`drift` / `deepen` / `branch` / `retool` / `modality_switch`); a `Pruner` kills low-scoring or near-duplicate children; dopamine backprops up the ancestor chain so a deep promising finding can rescue a meh-looking root. Per-session tree visualized in `journal/tree.md` as a Mermaid graph. Flat mode (`explore.py`) is unchanged.
- **Activities** *(v0.3)* — both `explore.py` and `wander.py` accept `--activities` and `--activity-mix` to choose what the wander *does* with each seed: research it, sketch code about it, design an app around it, riff visually, etc. Default is `research` (research-only, byte-identical to v0.2.1 behavior).
- **`program.md`** — the agent skill: what to do when invoked. Point Claude/Codex/Cursor here.
- **`dmn/`** — the boring stable plumbing: storage, embeddings, LLM providers, tools, importers. Don't modify unless you know why.

By design, exploration runs for a **fixed iteration or wall-clock budget**, defaulting to 12 iterations (~5 minutes-ish, in karpathy's spirit). The metric is **dopamine** — a weighted sum of alignment, novelty, surprise, and serendipity — and it's logged in the frontmatter of every brief so you can watch it evolve.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU. No API keys for the dry run.

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Bootstrap a taste profile (with a synthetic seed, no input needed)
uv run prepare.py --dry-run

# 4. Take a wander (no API keys; uses Wikipedia, DuckDuckGo, HN)
uv run explore.py --dry-run --iterations 2
```

That should leave you with a `data/dmn.sqlite` and two markdown briefs in `journal/`. If it didn't, something's broken — open an issue.

For the real thing:

```bash
# Real interview (cold start)
uv run prepare.py --interactive

# Or import from a Google Takeout dump
uv run prepare.py --import youtube,gmail,drive --takeout-dir ~/Downloads/Takeout
uv run prepare.py --import browser --browsers chrome,arc

# Real wander — default budget is 12 wall-clock minutes. Drop to 5 for a quick browse,
# push to 20-30 for a soak.
uv run explore.py
uv run explore.py --minutes 5
uv run explore.py --iterations 12   # reproducible iteration cap (alternative to --minutes)

# Tree-search mode (v0.2). Builds a tree of mutated seeds; promising deep findings
# back-prop up the ancestor chain. Output renders to `journal/tree.md` as a Mermaid graph.
uv run wander.py --iterations 12 --root-count 3 --max-depth 4
uv run wander.py --resume   # continue from the existing open frontier in SQLite
```

DMN's per-iteration cost is much higher than autoresearch's (multi-tool research + LLM
synthesis vs. a single training step), which is why the default budget is 12 minutes
instead of karpathy's 5. Empirically 10–15 minutes is the sweet spot.

## LLM providers

DMN talks to whichever LLM you point it at via `DMN_LLM_PROVIDER`. Set the corresponding
env vars and you're done — no config files.

| `DMN_LLM_PROVIDER` | env vars                                       | how to install                                                             |
| ------------------ | ---------------------------------------------- | -------------------------------------------------------------------------- |
| `anthropic`        | `ANTHROPIC_API_KEY`                            | `uv add anthropic` (already in deps)                                        |
| `openai`           | `OPENAI_API_KEY`                               | `uv add openai` (already in deps)                                           |
| `ollama`           | (none)                                          | `brew install ollama && ollama pull llama3.1:8b && ollama serve`            |
| `lmstudio`         | (none)                                          | download [LM Studio](https://lmstudio.ai), load a model, start local server |
| `stub`             | (none)                                          | always available; templated echo for offline dev                            |

`ollama` and `lmstudio` reuse the OpenAI SDK pointed at a local OpenAI-compatible
endpoint. Defaults: `ollama` → `http://localhost:11434/v1` with `llama3.1:8b`; `lmstudio`
→ `http://localhost:1234/v1` with `local-model`. Override either via `DMN_LLM_BASE_URL`
and `DMN_LLM_MODEL`. On startup DMN does a 2-second health-check against the base URL
and prints a friendly hint if the local server isn't running. If you don't set anything,
DMN auto-picks Anthropic if `ANTHROPIC_API_KEY` is set, then OpenAI, then falls back to
the stub.

For a fully-local run with no data leaving your laptop:

```bash
DMN_LLM_PROVIDER=ollama DMN_EMBEDDINGS=st uv run explore.py
```

## Activities — what does the mind wander on?

v0.3 turns the wander loop into a registry of pluggable **activities**. Each activity
takes a seed and returns a brief (markdown body) plus optional disk artifacts. Per-iter
the loop samples one activity from your `--activity-mix` (weighted), runs it, scores
the brief for dopamine, and writes it to the journal exactly like a research brief.

| activity            | what it does                                                          | needs                                  |
| ------------------- | --------------------------------------------------------------------- | -------------------------------------- |
| `research`          | The classic flow: search → score → synthesize a 5-bullet brief.        | (none — works with stub LLM)            |
| `code_sketch`       | LLM writes a ≤100-line stdlib+numpy Python sketch; optionally runs it.| (none; +`--execute` for sandbox run)    |
| `app_idea`          | 1-page markdown PRD + Mermaid architecture diagram + risks.            | (none)                                  |
| `algorithm_explore` | Picks an adjacent algorithm, writes a tiny demo + ASCII / matplotlib.  | (`matplotlib` extra; degrades to ASCII) |
| `ml_experiment`     | Tiny torch experiment, ≤200 lines, ≤30s on CPU, prints `METRIC: ...`.  | (`torch` extra)                         |
| `image_riff`        | LLM rewrites the seed as a visual prompt, calls an image backend.      | `GOOGLE_API_KEY` *or* `REPLICATE_API_TOKEN` |
| `music_riff`        | LLM rewrites the seed as musical direction, calls an audio backend.    | `STABILITY_API_KEY` (Stable Audio 2.0) or Lyria/Suno when public |
| `video_riff`        | Same shape as image; off by default in any sane mix (latency).         | `REPLICATE_API_TOKEN`                   |
| `mood_journal`      | Reflective ~200-word journal entry over your top clusters.             | (none)                                  |

Examples:

```bash
# Default: research-only, byte-identical to v0.2.1
uv run wander.py --iterations 12

# Code-curious: research + sketches + app PRDs, with sandboxed execution
uv run wander.py --activities research,code_sketch,app_idea --iterations 12 --execute

# Multimodal: research + image riffs (works with GOOGLE_API_KEY set)
uv run wander.py --activity-mix research:3,image_riff:1 --iterations 8

# The works
uv run wander.py \
  --activity-mix research:4,code_sketch:2,app_idea:1,algorithm_explore:1,image_riff:1,music_riff:1 \
  --execute --iterations 20
```

`--execute` is opt-in. It runs code-generating activities under `--sandbox auto`
(default: Docker when available, otherwise subprocess), `--sandbox docker`,
`--sandbox subprocess`, or `--sandbox none`. Subprocess mode strips all `*_API_KEY` /
`*_TOKEN` / `*_SECRET` env vars from the child, so even a hostile LLM-generated script
can't see your credentials, but Docker is stronger isolation when available. Run
`uv run python -c "from dmn.sandbox import run_python; ..."` if you want to test it.

`mutate_modality_switch` (v0.3) is a new tree-mode mutation that keeps the parent's
seed but switches the activity. So a research brief on "RLHF preference datasets" can
spawn a `code_sketch` child that implements a tiny preference-pair sampler — that's
the wandering-across-modalities pattern, in the tree. It only fires when
`--activities` / `--activity-mix` lists more than one activity.

## Multimodal: generative-media artifacts

DMN can optionally call generative-media tools (image, music, video) at the end of each
brief, embedding the result inline:

```bash
uv sync --extra generators
uv run explore.py --generate --modalities image,music
```

Backends in v0.1.1 (each module under `dmn/generators/` self-registers; pick one per
modality based on the env you have configured):

| modality | backend         | env var                                | notes                                  |
| -------- | --------------- | -------------------------------------- | -------------------------------------- |
| image    | `nano_banana`   | `GOOGLE_API_KEY` / `GEMINI_API_KEY`    | Gemini 2.5 Flash Image                 |
| image    | `replicate_image` | `REPLICATE_API_TOKEN`                | default model: `black-forest-labs/flux-schnell` (override via `DMN_IMAGE_MODEL_REPLICATE`) |
| music    | `stable_audio`  | `STABILITY_API_KEY`                    | Stable Audio 2.0 (`/v2beta/audio/...`) |
| music    | `lyria`         | `GOOGLE_API_KEY`                       | Lyria 2 — currently allow-listed; stub returns "unavailable" |
| music    | `suno`          | `SUNO_API_KEY`                         | Stub until upstream API is public      |
| video    | `replicate_video` | `REPLICATE_API_TOKEN`                | default model: `lucataco/animate-diff` (override via `DMN_VIDEO_MODEL_REPLICATE`) |
| video    | `video_stub`    | (none)                                  | placeholder; Veo 3 / Sora / Runway / Kling adapters drop in here |
| (any)    | `dryrun_*`      | (none)                                  | tiny placeholder files for `--dry-run` |

Generators are OFF by default to keep the loop fast and key-free. The artifact's own
embedding is **not** yet folded into the dopamine score — that's a v0.2 item (would need
CLIP for images, CLAP for audio). For now only the brief's text is scored.

Generated files land in `data/artifacts/` and are referenced from the brief's frontmatter
(`artifact: {...}`) and embedded inline in the markdown body (`![generated](...)` for
images, `[Listen](...)` for audio).

## Running as a skill

DMN ships as a one-skill repo: point your coding agent at `program.md` and it knows what to do.

**Claude / Cursor / Codex CLI**:

```
Hi, have a look at program.md and let's go for a wander.
```

The agent will check whether `data/dmn.sqlite` exists. If not, it kicks off `prepare.py`. If yes, it runs `explore.py`, then surfaces the top-3 highest-dopamine briefs from this session (with the score breakdown) so you can read them in chat.

A copy of the skill also lives at `.claude/skills/default-mode-network/SKILL.md` so Claude Code can discover it natively.

## Design choices

- **Small surface area.** `explore.py` remains the flat-loop playground, while stable plumbing lives under `dmn/`. Tree mode and activities now have their own modules, so deeper changes should include tests.
- **Fixed budget.** Each session runs for a fixed iteration count (default 12) or wall-clock minutes. This makes runs comparable and keeps the agent honest.
- **Dopamine, not loss.** The reward is a soft, multi-component score, computed in float space and logged with every brief:
  - `alignment` = max cosine similarity to your taste cluster centroids. Stuff you'd like.
  - `novelty` = `1 − max cosine similarity` to recent journal entries. Stuff you haven't seen.
  - `surprise` = how much more aligned to your taste than to your recent finds. *New angles* on the things you love.
  - `serendipity` = epsilon-greedy override: small chance to pursue a low-alignment, high-novelty item. Keeps your taste horizons drifting. Only fires when `fulfillment` meets a minimum floor — failed searches can't jackpot here.
  - `fulfillment` = did the research/tools actually return useful material? Research uses count + quality of tool hits, relevance caps, and a penalty when the LLM admits the search whiffed. Creation/media activities use a shared score over grounding, artifacts, execution, and body quality.
  - `total = α·alignment + β·novelty + γ·surprise + δ·fulfillment + ε·serendipity`. Constants live at the top of `dmn/taste.py` — tune them.
- **Multiple taste clusters, not a centroid.** Your interests are a polytopia. K-means over your interest embeddings preserves "I like Lisp AND fermentation AND polyrhythms", and the seed generator can sample one cluster *or* two distant ones (cross-pollination).
- **Online learning.** High-dopamine briefs nudge the nearest cluster centroid by a small step. Your taste profile drifts as you read.
- **Minimal stack.** SQLite + JSON-blob vectors + a tiny in-memory numpy index. No vector DB, no Postgres, no Docker. You can `cat` the database and read your own profile.
- **Graceful degrade everywhere.** No `ANTHROPIC_API_KEY`? You get the stub LLM. `sentence-transformers` not installed? Hash-based fallback embeddings. `arxiv` package missing? That tool just returns `[]`. The system runs.

## Profile portability

Your taste profile is just a SQLite + a small JSON. You can export it, swap with friends,
or merge it into a date-night shared profile. The vision is "swap profiles like mixtapes" —
share an anonymized taste-vector dump with a friend, see what their wandering loop surfaces
that yours wouldn't.

```bash
uv run profile.py export --out me.dmn.json
uv run profile.py export --anonymize --out me.anon.dmn.json    # strips raw text
uv run profile.py import friend.dmn.json --replace             # or --append
uv run profile.py merge friend.dmn.json --blend 0.5            # see note below
uv run eval.py unrated
uv run eval.py rate 42 5 --feedback "exactly the kind of rabbit hole I wanted"
uv run eval.py report
```

**Heads-up:** `merge` still produces a unioned profile (concat interests + concat
clusters), marks those clusters as incoherent, and expects you to import/recluster before
using it as a real shared profile. The `--blend` flag is parsed but currently ignored.

The export format (`profile.dmn.json`, schema v1) carries a per-model fingerprint so you
can't accidentally import a profile built from a different embedding model — that would
silently corrupt distances. Re-embedding-on-import is also a v0.2 item.

## Reading the journal

Each wander writes markdown briefs under `journal/` (for git/agent compatibility) and
regenerates HTML views for browsing: `dashboard.html` (run health, ratings, fulfillment),
`index.html` (all briefs by dopamine), `today.html` (last 24h rollup),
`tree.html` (wander session tree), plus one
`{slug}.html` per brief. Open `journal/index.html` in a browser, or rebuild anytime
with `uv run journal.py build`.

## Privacy

All profile data — your interview answers, browser history, embeddings, journal — lives **locally** in `data/` and `journal/`, both gitignored. None of it is uploaded anywhere.

The data that can leave your machine is whatever goes into an enabled remote LLM/API call. During normal wandering, DMN sends the seed question and public search results for planning/synthesis. During prepare/relabel, remote cluster labeling sends a small sample of cleaned interest text from each cluster so the LLM can name the theme. Use `uv run prepare.py --local-labels` / `uv run prepare.py --relabel-only --local-labels`, `DMN_LLM_PROVIDER=stub`, and local embeddings if you want nothing private sent to a remote model.

## Project structure

```
prepare.py            — one-time interview + importers + clustering (you can re-run)
explore.py            — the flat wandering loop (the agent edits this)
wander.py             — best-first tree-search wander mode (v0.2)
profile.py            — export / import / merge taste profiles across DMN installs
journal.py            — rebuild HTML journal / run reports
eval.py               — rate briefs and inspect dopamine-vs-rating alignment
program.md            — agent instructions
SKILL.md              — Anthropic skill format pointer
dmn/                  — plumbing
  taste.py            — profile, clusters, dopamine reward function
  seeds.py            — seed-question generators (cold/cluster/cross/drift/trending)
  labeling.py         — LLM-synthesized cluster themes + per-interest tags
  loop.py             — shared synthesis prompt + tool planner + tool runner (v0.2)
  tree.py             — Frontier / Pruner / UCB / backprop / mutation operators (v0.2/v0.3)
  sandbox.py          — auto / docker / subprocess code execution with env stripping
  embeddings.py       — sentence-transformers + openai backends + hash fallback
  llm.py              — anthropic / openai / ollama / lmstudio / stub providers
  store.py            — SQLite schema + helpers (v5 adds journal activity / artifacts / execution)
  journal.py          — markdown brief writer + index + today's notebook + tree.md (v0.2/v0.3)
  html_journal.py     — HTML journal views (index/today/tree + per-brief pages)
  portability.py      — schema-v5 profile.dmn.json serialization
  activities/         — pluggable wander activities (research/code/app/algorithm/ml/image/music/video/mood) (v0.3)
  tools/              — pluggable search backends (arxiv, wikipedia, ddg, hn, reddit, ...)
  generators/         — pluggable generative-media backends (nano-banana, replicate, stable-audio, ...)
  importers/          — browser history, takeout, twitter export, csv
data/                 — local-only sqlite + artifacts + caches (gitignored)
journal/              — generated briefs (gitignored)
pyproject.toml
```

## Knobs you'll want to turn

- `dmn/taste.py` — the dopamine constants (`ALPHA`, `BETA`, `GAMMA`, `DELTA`, `EPS`, `SERENDIPITY_BONUS`, `SERENDIPITY_MIN_FULFILLMENT`). Serendipity now has two independent knobs: `EPS` is *how often* it fires (probability) and `SERENDIPITY_BONUS` is *how much it's worth* when it does. Crank either up to widen your horizons; crank down to drill in. Raise `DELTA` to reward briefs backed by solid tool hits.
- `dmn/seeds.py` — the mix of seed strategies. The default `explore.py` samples them with fixed probabilities; tune those. Each `Seed` now carries a `query` (clean keywords handed to the search APIs) distinct from its `text` (the conversational question the LLM synthesizes against) — `search_query()` derives one from the other when a generator doesn't set it.
- `wander.py --explore` — UCB exploration weight for tree-mode frontier selection. `0` is pure best-first (greedy); higher values give shallower/less-committed branches an exploration premium so the wander doesn't tunnel down one thread.
- `explore.py` / `wander.py` — loop strategy. Try a different planner, scorer, synthesis prompt, or tree policy. Log a one-line rationale near the changed strategy.
- `DMN_LLM_PROVIDER` env var: `anthropic` (default if `ANTHROPIC_API_KEY` set), `openai`, or `stub`.
- `DMN_EMBEDDINGS` env var: `st` (default, sentence-transformers) or `openai`.

## Notable forks

(none yet — be the first)

## License

MIT. See `LICENSE`. Have fun. Don't be creepy with someone else's data.
