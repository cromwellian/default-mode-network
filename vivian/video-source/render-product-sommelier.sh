#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/vivian/the-product-sommelier-video.mp4"
PREVIEW="$ROOT/vivian/previews/the-product-sommelier-video-preview.png"
FRAMES_DIR="${TMPDIR:-/tmp}/dmn_product_sommelier_frames"
PYTHON_BIN="${PYTHON_BIN:-/Users/vivian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$ROOT/vivian/video-source/render_product_sommelier.py" \
  --out-dir "$FRAMES_DIR" \
  --fps 24 \
  --duration 30

ffmpeg -y \
  -framerate 24 -i "$FRAMES_DIR/frame_%04d.png" \
  -f lavfi -i "sine=frequency=82:sample_rate=48000:d=30" \
  -filter_complex "[1:a]volume=0.028,afade=t=in:st=0:d=2,afade=t=out:st=27.5:d=2.5[a]" \
  -map 0:v -map "[a]" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 96k -movflags +faststart \
  "$OUT"

ffmpeg -y -ss 00:00:12 -i "$OUT" -frames:v 1 -update 1 "$PREVIEW"

ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,avg_frame_rate -of json "$OUT"
