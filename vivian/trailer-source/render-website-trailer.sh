#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/vivian/vivian-website-trailer.mp4"
PREVIEW="$ROOT/vivian/previews/vivian-website-trailer-preview.png"
FRAMES_DIR="${TMPDIR:-/tmp}/dmn_vivian_website_trailer_frames"
PYTHON_BIN="${PYTHON_BIN:-/Users/vivian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

mkdir -p "$FRAMES_DIR"
"$PYTHON_BIN" "$ROOT/vivian/trailer-source/render_website_trailer.py" \
  --out-dir "$FRAMES_DIR" \
  --fps 24

ffmpeg -y \
  -framerate 24 -i "$FRAMES_DIR/frame_%04d.jpg" \
  -f lavfi -i "sine=frequency=48:sample_rate=48000:d=47" \
  -f lavfi -i "sine=frequency=96:sample_rate=48000:d=47" \
  -filter_complex "[1:a]volume=0.030[a0];[2:a]volume=0.018,afade=t=in:st=16:d=8[a1];[a0][a1]amix=inputs=2,afade=t=in:st=0:d=2,afade=t=out:st=42:d=4[a]" \
  -map 0:v -map "[a]" \
  -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p \
  -c:a aac -b:a 112k -movflags +faststart \
  "$OUT"

ffmpeg -y -ss 00:00:21 -i "$OUT" -frames:v 1 -update 1 "$PREVIEW"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name,width,height,avg_frame_rate -of json "$OUT"
