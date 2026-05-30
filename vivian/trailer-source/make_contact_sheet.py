#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="vivian/trailer-source/site-images")
    parser.add_argument("--out", default="/private/tmp/vivian-site-images-contact.jpg")
    args = parser.parse_args()

    files = sorted(Path(args.image_dir).glob("*"))
    cell_w = 300
    cell_h = 220
    cols = 5
    rows = max(1, math.ceil(len(files) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 238, 225))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)

    for index, path in enumerate(files):
        with Image.open(path) as source:
            size = source.size
            image = source.convert("RGB")
        image.thumbnail((cell_w - 18, 164))
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 8))
        draw.text((x + 10, y + 176), f"{index:02d} {path.name[:30]}", fill=(20, 20, 20), font=font)
        draw.text((x + 10, y + 197), f"{size[0]} x {size[1]}", fill=(90, 80, 70), font=small)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(out)
    print(f"{len(files)} images")


if __name__ == "__main__":
    main()
