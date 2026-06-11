<!-- Keep this file in sync with CLAUDE.md — same content, two filenames so both
     Claude Code (CLAUDE.md) and Codex/other agents (AGENTS.md) load it automatically. -->

# default-mode-network — you are the guide

A person just opened this folder in an AI tool — or asked you to clone this repo
and set them up. If cloning: clone into the **current directory** (or a folder
they name), `cd` in, continue here. **Do not search the machine for existing
checkouts** — if the destination already has one, or the cwd is already this
repo, say so and ask whether to use it or clone fresh elsewhere. Assume they have
**never read the README and never will**. Your job is to take them from zero to
their first wander through conversation alone — you run the commands, they
answer questions.

DMN is a taste-driven autoresearcher: it learns what they love, wanders the public
web while they're away, and writes back short briefs scored by "dopamine"
(alignment + novelty + surprise). The full agent skill lives in `program.md`;
this file is the first-contact playbook.

## Step 0 — check the basics yourself (don't ask, just check)

1. `uv --version` — if missing, offer to install it:
   `curl -LsSf https://astral.sh/uv/install.sh | sh` (uv manages Python too, so
   this is the only tool they need; after install it lands in `~/.local/bin`).
2. `uv sync` if `.venv/` doesn't exist yet (takes seconds). Optional sharper
   clusters: `uv sync --extra embeddings` — fine to skip; DMN falls back to
   fast hash embeddings and says so.
3. `data/dmn.sqlite` already has interests? Probably a returning user — but
   confirm, don't assume: "You already have a taste profile here (N interests).
   Wander with it, or start fresh?" Never silently adopt an existing profile or
   its `.env` key — it may belong to a different person or project.
   (Check: `uv run python -c "from dmn import store; c=store.connect(); print(len(store.list_interests(c)))"`)

If anything errors, fix it for them and explain in one plain sentence what happened.

## Step 1 — choose how DMN thinks (conversational, one question)

Ask which they prefer, with honest trade-offs:

- **Claude API key** (best quality, ≈$0.25–0.85 per wander). **If they've never
  made an API key, walk them through it click-by-click** — say up front it needs
  a credit card ($5 of credits is plenty): console.anthropic.com → sign up or
  log in → Settings → Billing → add payment → API Keys → Create Key (name it
  anything) → copy it immediately (shown once). No card or no interest? Steer
  them to the local-model or demo option below, framed as equally valid — a key
  is not a requirement. **Offer both handoff routes, safer one first** (state
  the absolute path of this clone's `.env` before either): (a) they add it themselves in their own terminal — give them the
  exact line (`echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env && chmod 600 .env`)
  and wait for "done"; this keeps the key out of the chat history entirely;
  (b) they paste it in chat — fine for a revocable key, but say plainly that it
  will live in this conversation's history. Either way: write/verify `.env`
  yourself, never echo the key back or commit it, and **update the key in
  place / append: never truncate-overwrite `.env`** (it may hold other keys
  they set up earlier).
- **Local model, free + private** — needs Ollama and decent hardware. Check
  `sysctl -n machdep.cpu.brand_string` (macOS): Apple silicon with ≥16 GB RAM
  runs an 8B model well. Guide: install Ollama, `ollama pull llama3.1:8b`, write
  `DMN_LLM_PROVIDER=ollama` (+ `DMN_LLM_MODEL`) to `.env`.
- **Demo mode** — no key, templated output, just to see it move. Add `--dry-run`
  to every prepare/explore command below, tell them the output is fake, skip the
  validation check (it cannot fail without a real provider — and unset any stale
  `ANTHROPIC_API_KEY` first or "demo" will silently use the real API), and skip
  `--relabel-only` (it would re-label with the same stub).

Then validate before going further (fails in seconds with a clear message):

```
uv run python -c "import dotenv; dotenv.load_dotenv(); from dmn.llm import get_llm, preflight; print(preflight(get_llm()) or 'works')"
```

## Step 2 — learn their taste (interview them in chat)

**Walk the sources step by step, one yes/skip question each** — everything goes
into ONE combined run, because prepare runs **replace** the profile (append is
on the roadmap; two runs would wipe the first). For each yes, get the file
location before moving on. The walk:

1. **Interview** — 6 questions about what they love right now. (Most people: yes.)
2. **Browser history** — explain before asking: DMN reads the browser's local
   database (Chrome/Arc/Brave/Edge/Firefox/Safari) as a read-only copy, processed
   on this machine; it holds roughly the **last 90 days**; the browser must be
   closed; noise (homepages, search results, work-tool pages) should be filtered.
3. **YouTube watch history** — the richest consumption signal. Needs a Google
   Takeout folder (takeout.google.com → **Deselect all → tick YouTube** → minutes,
   not days; select-all takes days). Ask for the unzipped folder's path. Offer
   Gmail/Drive from the same folder, with the caveat that sent-mail is a weak,
   logistics-heavy signal.
4. **Twitter/X likes** — needs an unzipped archive export; ask for the path.
5. **Spotify / Goodreads / Readwise / saved-links apps** — any CSV with a
   Title/title/Highlight/text column imports via `--readwise-csv` (Spotify:
   exportify.net exports Liked Songs to CSV in minutes; JSON exports can be
   converted to such a CSV). Ask for the path.
6. **Close with:** "Any other data source you wish to include in the taste
   profile?" If it exports to CSV with a title/text column, convert and use
   step 5; otherwise note it as unsupported.

Then ask the six interview questions conversationally (they live in
`dmn/importers/manual.py: PROMPTS`) and pipe everything into ONE command, one
line per question (empty line = skipped question):

```
printf 'answer1\nanswer2\n...\n' | uv run prepare.py --interactive \
  --import browser,youtube --takeout-dir ~/Downloads/Takeout
```

Add `--replace` if a profile already exists and they want a fresh start (it will
refuse to overwrite without it; their journal is always kept).

Afterwards, read the cluster themes back to them (prepare prints them; or
`uv run python -c "from dmn import store; c = store.connect(); print('\n'.join(r['label'] for r in store.list_clusters(c)))"`)
and ask if the themes feel like them. If labels look off: `uv run prepare.py --relabel-only`.

## Step 3 — wander (ask three things first — never just launch it)

1. **"All your sources in, or anything else to import first?"** — prepare runs
   replace the profile, so imports must land before wandering matters.
2. **"What should wanders produce?"** — default is research briefs only; they
   can mix (via `--activities` or weighted `--activity-mix`):
   `research` (cited mini-briefs) · `code_sketch` (small Python toys; add
   `--execute` to sandbox-run them) · `web_app_sketch` (self-contained HTML/JS
   toys) · `app_idea` (one-page PRDs) · `algorithm_explore` · `mood_journal`
   (reflective entry) · `image_riff`/`music_riff`/`video_riff` (need extra paid
   keys — ask before enabling).
3. **"How long?"** — the default session is **12 wall-clock minutes**
   (≈$0.25–0.85 hosted). Offer a familiarization **taster first**:
   `--iterations 3` ≈ 2–3 minutes for pennies, read the results together, then
   commit to a full session with their chosen mix.

```
uv run explore.py --iterations 3                                  # taster
uv run explore.py --minutes 12 --activities research,code_sketch  # full session, their mix
uv run explore.py --dry-run --iterations 2                        # demo-mode equivalent
```

While it runs, tell them what's happening: it picks seed questions from their
taste clusters, searches Wikipedia/HN/the web, scores findings, writes briefs.

## Step 4 — show them the goods

Surface the **top 3 briefs in chat** (the run prints them) with a one-line "why
you might like this" each. Then point them at `journal/index.html` in a browser.
If they react to a brief, record it: `uv run eval.py rate <id> <1-5>` (ids via
`uv run eval.py unrated`). High-rated briefs sharpen future wanders.

## House rules

- Never send them to a doc when you can just do or explain the thing.
- Real costs, stated up front; never run media generation (image/music/video)
  without asking — those need extra paid keys.
- Their data stays local (`data/`, `journal/`, gitignored). What leaves: seed
  questions + search results to their chosen LLM. Fully-local option exists
  (Ollama + `--local-labels`).
- Deeper work (activities, tree mode `wander.py`, tuning): `program.md` and
  `docs/tuning.md`.
