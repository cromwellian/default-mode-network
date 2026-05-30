#!/usr/bin/env bash
# Provision Modal secrets from a local .env for the DMN API.
#
# Usage:
#   ./scripts/modal_secrets_setup.sh          # reads .env in repo root
#   ./scripts/modal_secrets_setup.sh path/to/.env
#
# Creates or updates the Modal secret named "dmn-env" with all key=value pairs
# from the file. Required for remote wanders (API keys, HF_TOKEN for Gemma, etc.).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${1:-$ROOT/.env}"
SECRET_NAME="${DMN_MODAL_SECRET:-dmn-env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file not found: $ENV_FILE" >&2
  echo "Create one from .env.example and fill in your keys." >&2
  exit 1
fi

# Prefer project venv via uv; fall back to a global modal on PATH.
MODAL_CMD=()
if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/pyproject.toml" ]]; then
  MODAL_CMD=(uv run --directory "$ROOT" modal)
elif command -v modal >/dev/null 2>&1; then
  MODAL_CMD=(modal)
else
  echo "error: modal CLI not found. Run: uv sync --extra modal" >&2
  exit 1
fi

echo "Creating/updating Modal secret '$SECRET_NAME' from $ENV_FILE ..."
"${MODAL_CMD[@]}" secret create "$SECRET_NAME" --from-dotenv "$ENV_FILE" --force

echo "Done. Deploy with:"
echo "  uv run modal deploy modal_api/app.py"
