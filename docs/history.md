# Version notes & upgrade guides

Design history that used to live at the top of the README. Useful when upgrading an
existing install or understanding why a subsystem looks the way it does. New users
don't need any of this.

## How labels are made (v0.1.2)

DMN's seed quality lives or dies on its cluster labels. Done badly, the seed generator
ends up asking "What's a fresh angle on YouTube (https://www.youtube.com/?" — which is
what happened in v0.1.0 when labels were just the nearest-centroid interest's raw text.
v0.1.2 fixed it in two places:

1. **Browser-importer cleanup** (`dmn/importers/browser.py`). Before any clustering, the
   importer drops homepage visits, login pages, generic-domain titles ("YouTube",
   "GitHub", "Gmail", …), search-result pages, very-short titles, and per-host floods
   (max 5 entries per host). Pass `--keep-noise` to bypass.
2. **Synthesized cluster labels** (`dmn/labeling.py`). After clustering in embedding
   space, DMN picks the 12 nearest-medoid members per cluster and asks the configured
   LLM for `{"theme": "...", "subtopics": [...], "rationale": "..."}` — a *thematic*
   label, not a verbatim title. The theme is stored as the cluster's `label` column
   (back-compat); the subtopics and rationale go into a `meta` JSON column. Subtopics
   flow through to seed prompts.

If a cluster's theme looks off, run `uv run prepare.py --relabel-only` to re-synthesize
labels against the existing profile (no re-import, no re-cluster — fast). Use
`--local-labels` to force the deterministic local labeler so sampled interest text is
not sent to a remote LLM during prepare/relabel.

## Clustering (v0.4)

DMN defaults to **HDBSCAN** on L2-normalized embeddings with **medoid** cluster
representatives (not K-means centroids). Sample weights combine `log1p(visit_count)`
with exponential recency decay; manual and interview interests are pinned as
high-recency seeds. Browser history is weighted by visit frequency (log-scale), so
frequently-visited sites have more influence on cluster centroids.

```bash
uv run prepare.py --cluster-method hdbscan   # default
uv run prepare.py --cluster-method kmeans    # legacy K-means (k=8)
uv run prepare.py --cluster-method gmm       # Gaussian mixture + BIC k-selection

# Tune recency half-life (days) and HDBSCAN min cluster size
uv run prepare.py --recency-half-life 90 --min-cluster-size 20
```

**Noise bucket:** points HDBSCAN cannot assign land in an `ambient / unclustered`
cluster (`is_noise=1`). Seed sampling skips this bucket unless a small serendipity
roll hits (~5%).

Inspect clusters (medoid text, method, noise flag, silhouette):

```bash
uv run scripts/inspect_clusters.py --top 10
```

## Wander controls and web apps (v0.5)

Tree-mode wander records a `run_id` per invocation, stores per-node visit/expansion
metrics in SQLite, and keeps `journal/tree.html` focused on the latest run when session
data is available. Frontier can be bounded with beam pruning and stopped when new
expansions stop improving dopamine — see [tuning.md](tuning.md) for the flags.

Cross-pollination samples cluster pairs from a moderate distance band instead of always
forcing the farthest pair. Image and music riffs are strongly downweighted unless the
seed or cluster text suggests visual/audio form would help, or you explicitly request
only that activity.

`web_app_sketch` (opt-in activity) writes a self-contained
`data/artifacts/web_app_sketch/<slug>/index.html` plus `manifest.json`; journal briefs
link to the app and embed it in a sandboxed iframe.

## Upgrading from earlier dev versions

Migrations live in `scripts/` and are idempotent — run in order if you're jumping
multiple versions:

| you ran      | run                                  | what it does                                                         |
| ------------ | ------------------------------------ | -------------------------------------------------------------------- |
| v0.1.0/0.1.1 | `uv run scripts/migrate_v0_1_2.py`   | re-filters noisy `browser:*` interests, re-clusters, re-labels       |
| v0.1.2       | `uv run scripts/migrate_v0_1_3.py`   | schema bump (journal activity/artifact columns)                      |
| v0.1.3       | `uv run scripts/migrate_v0_1_4.py`   | schema bump (run/session bookkeeping)                                |
| ≤ v0.3       | `uv run scripts/migrate_v0_4.py`     | HDBSCAN/medoid upgrade for existing profiles                         |
