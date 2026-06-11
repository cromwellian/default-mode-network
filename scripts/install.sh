#!/bin/sh
# default-mode-network one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/cromwellian/default-mode-network/main/scripts/install.sh | sh
#
# Checks git, installs uv if missing (uv brings its own Python), clones the
# repo, installs dependencies, and launches the guided setup wizard.
set -eu

TARGET="${1:-default-mode-network}"
REPO="https://github.com/cromwellian/default-mode-network"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. macOS: run 'xcode-select --install' and re-run this script."
  echo "Linux: install git with your package manager (e.g. 'sudo apt install git')."
  exit 1
fi

UV="uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (the only tool DMN needs - it manages Python for you)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin, which may not be on PATH yet.
  UV="$HOME/.local/bin/uv"
  command -v "$UV" >/dev/null 2>&1 || UV="$HOME/.cargo/bin/uv"
  if ! command -v "$UV" >/dev/null 2>&1; then
    echo "uv installed but not found on PATH - open a new terminal and re-run this script."
    exit 1
  fi
fi

if [ -e "$TARGET" ]; then
  echo "'$TARGET' already exists here - cd into it and run: uv run dmn-setup"
  exit 1
fi

echo "Cloning into $TARGET..."
git clone --quiet "$REPO" "$TARGET"
cd "$TARGET"
echo "Installing dependencies..."
"$UV" sync --quiet

# Under `curl | sh`, stdin is the script pipe even in a real terminal, so probe
# /dev/tty (the controlling terminal) and hand it to the interactive wizard.
if [ -t 0 ]; then
  exec "$UV" run dmn-setup
elif [ -t 1 ] && [ -r /dev/tty ]; then
  exec "$UV" run dmn-setup < /dev/tty
fi
echo
echo "Done. Next steps:"
echo "  cd $TARGET"
echo "  uv run dmn-setup        # guided setup (or open this folder in Claude Code / Codex)"
