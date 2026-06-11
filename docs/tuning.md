# Tuning DMN

Everything beyond the defaults: dopamine constants, seed strategies, wander-tree
parameters, providers, embeddings, and generative-media backends.

## Knobs you'll want to turn

- `dmn/taste.py` — the dopamine constants (`ALPHA`, `BETA`, `GAMMA`, `DELTA`, `EPS`,
  `SERENDIPITY_BONUS`, `SERENDIPITY_MIN_FULFILLMENT`). Serendipity has two independent
  knobs: `EPS` is *how often* it fires (probability) and `SERENDIPITY_BONUS` is *how much
  it's worth* when it does. Crank either up to widen your horizons; crank down to drill
  in. Raise `DELTA` to reward briefs backed by solid tool hits.
- `dmn/seeds.py` — the mix of seed strategies. The default `explore.py` samples them with
  fixed probabilities; tune those. Each `Seed` carries a `query` (clean keywords handed
  to the search APIs) distinct from its `text` (the conversational question the LLM
  synthesizes against) — `search_query()` derives one from the other when a generator
  doesn't set it.
- `explore.py` / `wander.py` — loop strategy. Try a different planner, scorer, synthesis
  prompt, or tree policy. Log a one-line rationale near the changed strategy.

## Wander-tree parameters

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

| flag | default | effect |
| ---- | ------- | ------ |
| `--root-count` | 3 | independent root seeds per session |
| `--max-depth` | 4 | how deep a chain of mutations can go |
| `--children-per-expansion` | 3 | children spawned per expanded node |
| `--beam-width` | 0 (off) | keep only the top-N open frontier leaves after each expansion |
| `--patience` | 0 (off) | restart from a fresh root after N expansions with no improvement |
| `--min-improvement` | 0.01 | dopamine gain below this counts as "no improvement" for patience |
| `--margin`, `--min-absolute` | 0.05 / 0.15 | pruner thresholds: children too far below the best, or simply too weak, are killed |
| `--similarity-threshold` | 0.95 | near-duplicate children (cosine vs siblings/ancestors) are pruned |
| `--restart-prob` | 0.08 | chance per iteration to start a fresh cluster-seeded root |
| `--explore` | 0.05 | UCB exploration weight for frontier selection; `0` = pure best-first |
| `--until-dopamine` | off | stop once any brief reaches this score |
| `--ground/--no-ground` | on | creation activities (code/app/web sketches) first gather external references |
| `--report/--no-report` | on | NotebookLM-style narrated run report |

`--code-budget small|medium|large` bounds generated-code size and sandbox timeout
(30s / 90s / 120s); `--execute` opts into running sketches under
`--sandbox auto|docker|subprocess|none` (subprocess mode strips `*_API_KEY` /
`*_TOKEN` / `*_SECRET` from the child env).

`mutate_modality_switch` is a tree-mode mutation that keeps the parent's seed but
switches the activity — a research brief on "RLHF preference datasets" can spawn a
`code_sketch` child implementing a tiny preference-pair sampler. It only fires when
`--activities` / `--activity-mix` lists more than one activity.

Legacy flat-mode media generation (predates the riff activities) still works:
`uv sync --extra generators && uv run explore.py --generate --modalities image,music`
appends generated media to each brief.

## LLM providers

Set `DMN_LLM_PROVIDER`, or let DMN auto-pick: Anthropic if `ANTHROPIC_API_KEY` is set,
then OpenAI, then the stub.

| `DMN_LLM_PROVIDER` | env vars            | how to install                                                              |
| ------------------ | ------------------- | --------------------------------------------------------------------------- |
| `anthropic`        | `ANTHROPIC_API_KEY` | already in deps                                                              |
| `openai`           | `OPENAI_API_KEY`    | already in deps                                                              |
| `ollama`           | (none)              | `brew install ollama && ollama pull llama3.1:8b` — or `uv run dmn-setup`     |
| `lmstudio`         | (none)              | download [LM Studio](https://lmstudio.ai), load a model, start local server  |
| `vllm`             | (none)              | point `DMN_LLM_BASE_URL` at a vLLM OpenAI-compatible endpoint                |
| `stub`             | (none)              | always available; templated echo for offline dev                            |

Local providers reuse the OpenAI SDK against a local endpoint. Defaults: `ollama` →
`http://localhost:11434/v1` with `llama3.1:8b`; `lmstudio` → `http://localhost:1234/v1`
with `local-model`; `vllm` → `http://localhost:8000/v1`. Override with
`DMN_LLM_BASE_URL` / `DMN_LLM_MODEL`. On startup DMN health-checks the base URL,
validates your key/model with one tiny call, and prints a cost estimate.

### Using a gateway (Vercel AI Gateway, OpenRouter)

Any OpenAI-compatible gateway works through the `openai` provider — one key,
many models, provider-side fallbacks and spend dashboards:

```bash
DMN_LLM_PROVIDER=openai \
OPENAI_API_KEY=<your-gateway-key> \
DMN_LLM_BASE_URL=https://ai-gateway.vercel.sh/v1 \
DMN_LLM_MODEL=anthropic/claude-sonnet-4-6 \
uv run explore.py --iterations 3
```

(OpenRouter: `DMN_LLM_BASE_URL=https://openrouter.ai/api/v1`, model ids like
`anthropic/claude-sonnet-4-6`.) The startup preflight validates the gateway key
and model with one tiny call, same as a direct provider. The default newcomer
path remains one Anthropic key — gateways are bring-your-own.

**Caveat:** `DMN_LLM_BASE_URL` is shared with the local providers (it's also how
you point `ollama`/`lmstudio` at a custom host). Set it per-run as above rather
than persisting it in `.env`, or a later `DMN_LLM_PROVIDER=ollama` run would be
aimed at your gateway. The preflight catches the breakage in seconds either way.

## Embeddings

`DMN_EMBEDDINGS`: `st` (default; local sentence-transformers, needs
`uv sync --extra embeddings` — if that fails on an Apple-silicon Mac with a
Rosetta/x86 toolchain (`platform.machine()` says x86_64 but `sysctl -n
machdep.cpu.brand_string` says Apple), pin a native interpreter first:
`uv python pin cpython-3.13-macos-aarch64-none`, then re-sync), `openai` (`text-embedding-3-small`, needs
`OPENAI_API_KEY`), or `hash` (no-dependency deterministic fallback). When `st` is
requested but not installed, DMN falls back to `hash` with a one-time notice. Profiles
are stamped with the backend that embedded them; explore/wander refuse to run when the
current backend doesn't match (mixing embedding spaces silently corrupts every score).

## Generative-media backends

Modules under `dmn/generators/` self-register; DMN picks per modality based on which
env vars are set:

| modality | backend           | env var                                                  | notes                                          |
| -------- | ----------------- | -------------------------------------------------------- | ---------------------------------------------- |
| image    | `nano_banana`     | `GOOGLE_API_KEY` / `GEMINI_API_KEY`                       | Gemini 2.5 Flash Image; preferred when present |
| image    | `replicate_image` | `REPLICATE_API_TOKEN`                                     | default `black-forest-labs/flux-schnell`; override `DMN_IMAGE_MODEL_REPLICATE` |
| image    | (HF fallback)     | `HF_TOKEN`                                                | override model via `DMN_IMAGE_MODEL_HF`        |
| music    | `stable_audio`    | `STABILITY_API_KEY`                                       | Stable Audio 2.0                               |
| music    | `lyria`           | `GOOGLE_API_KEY`                                          | allow-listed; stub returns "unavailable"       |
| music    | `suno`            | `SUNO_API_KEY`                                            | stub until upstream API is public              |
| video    | `runway_video`    | `RUN_API_KEY` / `RUNWAYML_API_SECRET` / `RUNWAY_API_KEY`  | default `seedance2`, 10s clips                 |
| video    | `replicate_video` | `REPLICATE_API_TOKEN`                                     | default `lucataco/animate-diff`; override `DMN_VIDEO_MODEL_REPLICATE` |
| (any)    | `dryrun_*`        | (none)                                                    | tiny placeholder files for `--dry-run`         |

Image riffs prefer `nano_banana` exclusively when configured (better at diagram text);
set `DMN_IMAGE_ALLOW_FALLBACKS=1` to allow HF/Replicate fallbacks after errors. Video
riffs are disabled unless `DMN_ENABLE_VIDEO_RIFFS=1` (slow + expensive); Runway knobs:
`DMN_VIDEO_RUNWAY_MODEL` (seedance2 supports 5–15s), `DMN_VIDEO_SECONDS`,
`DMN_VIDEO_RUNWAY_RATIO`, `DMN_VIDEO_RUNWAY_TIMEOUT`, `DMN_VIDEO_RUNWAY_POLL_SECONDS`.

Generated files land in `data/artifacts/` and are referenced from each brief's
frontmatter and body. The artifact's own embedding is not yet folded into the dopamine
score — that needs CLIP/CLAP and is future work.
