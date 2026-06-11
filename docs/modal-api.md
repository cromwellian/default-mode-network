# Modal REST API

Run DMN wanders remotely on [Modal](https://modal.com) with profile upload and artifact
URLs. All profiles and run artifacts are scoped under a **caller-provided `user_id`**
(your app's tenant key). This is for app builders — nothing here is needed for local use.

## Setup

```bash
cp .env.example .env               # fill in ANTHROPIC_API_KEY, HF_TOKEN, etc.
./scripts/modal_secrets_setup.sh   # loads .env → Modal secret "dmn-env"
uv sync --extra modal
uv run modal deploy modal_api/app.py
```

The deploy prints a URL like `https://your-workspace--default-mode-network-api.modal.run`.

## Provision a profile

Prepare locally, then upload the SQLite file under your user id:

```bash
USER_ID=alice
uv run prepare.py --dry-run     # or your real import flow
curl -X POST "$API/v1/users/$USER_ID/profiles" -F "file=@data/dmn.sqlite"
# → {"user_id": "alice", "profile_id": "abc123...", "stats": {...}}
```

`user_id` must be 1–64 chars, start with alphanumeric, and contain only letters,
digits, `.`, `_`, or `-`.

## Trigger a wander

All `wander.py` flags are supported in the JSON body. Use `llm.preset` for Modal-hosted
vLLM (Qwen, Gemma; set `VLLM_MODEL` to change the default served model):

```bash
curl -X POST "$API/v1/users/$USER_ID/profiles/$PROFILE_ID/wander" \
  -H "Content-Type: application/json" \
  -d '{
    "iterations": 3,
    "root_count": 2,
    "max_depth": 2,
    "llm": {"provider": "vllm", "preset": "qwen2.5-7b"}
  }'
```

Remote Anthropic/OpenAI (from the Modal secret):

```bash
curl -X POST "$API/v1/users/$USER_ID/profiles/$PROFILE_ID/wander" \
  -H "Content-Type: application/json" \
  -d '{"iterations": 1, "root_count": 1, "max_depth": 1}'
```

## Response format

Wander responses include:

- `user_id`, `profile_id`, `run_id`
- `briefs` — journal rows with parsed `body_md` and frontmatter
- `artifacts` — each file with inline `content` (text/small images) or metadata only
- `urls` — map of relative path → fetch URL under
  `/v1/users/{user_id}/profiles/{id}/runs/{run_id}/files/...`

List vLLM presets: `GET /v1/models`. Document wander params: `GET /v1/wander/params`.
