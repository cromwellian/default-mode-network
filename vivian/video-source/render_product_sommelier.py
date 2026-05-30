#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1920
HEIGHT = 1080

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

PAPER = (244, 234, 216)
PAPER_WARM = (255, 248, 234)
INK = (21, 24, 25)
SOFT_INK = (56, 43, 35)
MUTED = (107, 101, 91)

SCENES = [
    {
        "start": 0.0,
        "end": 5.0,
        "label": "LOCAL DMN PROFILE / 139 SIGNALS",
        "title": "The Product\nSommelier",
        "body": "A tasting menu for interfaces with a palate.",
        "meta": "websites / browser signals / handmade artifacts",
        "accent": (155, 63, 46),
        "kind": "title",
    },
    {
        "start": 5.0,
        "end": 10.0,
        "label": "AI-NATIVE PRODUCT",
        "title": "The Second\nVoice",
        "body": "A research copilot served in a coupe chilled with skepticism.",
        "meta": "product / design / code blur",
        "accent": (35, 106, 93),
        "kind": "product",
    },
    {
        "start": 10.0,
        "end": 15.0,
        "label": "SPIRITS LANGUAGE",
        "title": "Latency\nNegroni",
        "body": "Bitter, exact, and useful for dashboards that pretend time is free.",
        "meta": "burnt orange / observability / cost",
        "accent": (180, 130, 45),
        "kind": "spirits",
    },
    {
        "start": 15.0,
        "end": 20.0,
        "label": "DOCUMENTARY ATTENTION",
        "title": "Markets are\ninterfaces",
        "body": "They teach scanning before reading, trust before explanation.",
        "meta": "Varanasi / Rajasthan / Nakasendo",
        "accent": (40, 95, 135),
        "kind": "travel",
    },
    {
        "start": 20.0,
        "end": 25.0,
        "label": "SLOW VISUAL PRACTICE",
        "title": "Plein-Air\nDashboard",
        "body": "An energy tool painted outside before it is specified inside.",
        "meta": "linseed / kilowatts / consequence",
        "accent": (96, 69, 125),
        "kind": "painting",
    },
    {
        "start": 25.0,
        "end": 30.0,
        "label": "OUTPUT ARTIFACT",
        "title": "Interfaces\nwith a palate",
        "body": "Generated locally from product, spirits, photography, travel, and paint.",
        "meta": "Vivian Cromwell / default-mode-network",
        "accent": (155, 63, 46),
        "kind": "final",
    },
]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if not path.exists():
        path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    return ImageFont.truetype(str(path), size=size)


FONTS = {
    "label": font("Arial Bold.ttf", 31),
    "meta": font("Arial Bold.ttf", 30),
    "body": font("Georgia.ttf", 50),
    "title": font("Georgia Bold.ttf", 145),
    "title_small": font("Georgia Bold.ttf", 122),
    "micro": font("Arial.ttf", 24),
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def ease(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def fade_alpha(t: float, start: float, end: float, fade: float = 0.45) -> float:
    return min(clamp((t - start) / fade), clamp((end - t) / fade))


def rgba(color: tuple[int, int, int], alpha: float = 1.0) -> tuple[int, int, int, int]:
    return (*color, int(255 * clamp(alpha)))


def mix(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = clamp(amount)
    return tuple(int(x + (y - x) * amount) for x, y in zip(a, b))


def active_scene(t: float) -> dict:
    for scene in SCENES:
        if scene["start"] <= t < scene["end"]:
            return scene
    return SCENES[-1]


def draw_rule(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], alpha: float) -> None:
    draw.line((x1, y1, x2, y2), fill=rgba(color, alpha), width=2)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        trial = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), trial, font=font_obj)[2]
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)
    return "\n".join(lines)


def draw_background(draw: ImageDraw.ImageDraw, t: float, scene: dict) -> None:
    accent = scene["accent"]
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=rgba(PAPER, 1))
    draw.rectangle((70, 64, 1850, 1016), outline=rgba(INK, 0.9), width=3)
    draw.rectangle((92, 86, 1828, 994), outline=rgba(INK, 0.16), width=1)

    for i, x in enumerate(range(120, 1360, 155)):
        alpha = 0.025 + 0.015 * math.sin(t * 0.9 + i)
        draw_rule(draw, x, 96, x + 86, 96, INK, alpha)
        draw_rule(draw, x, 984, x + 86, 984, INK, alpha)

    panel_x = 1412
    draw.rectangle((panel_x, 128, 1676, 782), fill=rgba(PAPER_WARM, 0.9))
    draw.rectangle((panel_x, 128, 1676, 782), outline=rgba(INK, 0.45), width=2)

    if scene["kind"] != "final":
        palette = [(155, 63, 46), (35, 106, 93), (40, 95, 135)]
        for idx, color in enumerate(palette):
            y = int(210 + idx * 96 + 22 * math.sin(t * (1.55 + idx * 0.18) + idx))
            draw.rectangle((1480 + idx * 92, y, 1548 + idx * 92, y + 270), fill=rgba(color, 0.74))

    progress = clamp(t / 30.0)
    draw.rectangle((70, 1013, int(70 + 1780 * progress), 1016), fill=rgba(INK, 0.9))
    draw.rectangle((70, 1018, int(70 + 1780 * progress), 1024), fill=rgba(accent, 0.75))


def draw_scene_motif(draw: ImageDraw.ImageDraw, t: float, scene: dict, p: float, alpha: float) -> None:
    accent = scene["accent"]
    kind = scene["kind"]
    phase = ease(p)

    if kind == "title":
        for idx, word in enumerate(["PRODUCT", "SPIRITS", "PHOTO", "TRAVEL", "PAINT"]):
            y = 575 + idx * 43
            draw.text((1450, y), word, font=FONTS["micro"], fill=rgba(INK, alpha * 0.62))
            draw.rectangle((1605, y + 9, 1605 + int(84 * (0.25 + idx * 0.12)), y + 18), fill=rgba(accent, alpha * 0.5))

    elif kind == "product":
        for idx in range(6):
            y = 185 + idx * 77
            w = 120 + (idx % 3) * 44
            draw.rectangle((1450, y, 1450 + w, y + 34), outline=rgba(INK, alpha * 0.38), width=2)
            draw.rectangle((1458, y + 9, 1458 + int(w * (0.3 + 0.5 * abs(math.sin(t + idx)))), y + 18), fill=rgba(accent, alpha * 0.55))
        draw.line((1452, 674, 1640, 674), fill=rgba(accent, alpha * 0.62), width=6)
        draw.text((1452, 700), "AGENT AS MATERIAL", font=FONTS["micro"], fill=rgba(INK, alpha * 0.55))

    elif kind == "spirits":
        cx = 1545
        cy = 430
        draw.ellipse((cx - 92, cy - 74, cx + 92, cy + 32), outline=rgba(INK, alpha * 0.62), width=5)
        draw.line((cx, cy + 32, cx, cy + 230), fill=rgba(INK, alpha * 0.62), width=5)
        draw.line((cx - 94, cy + 230, cx + 94, cy + 230), fill=rgba(INK, alpha * 0.62), width=5)
        draw.arc((cx - 120, cy - 112, cx + 36, cy + 44), 293, 77, fill=rgba(accent, alpha * 0.82), width=16)
        draw.multiline_text((1456, 698), "BITTER\nEXACT\nUSEFUL", font=FONTS["micro"], spacing=4, fill=rgba(INK, alpha * 0.55))

    elif kind == "travel":
        for idx in range(4):
            x = 1448 + (idx % 2) * 96
            y = 182 + (idx // 2) * 128
            sky = mix(PAPER, accent, 0.18 + 0.1 * idx)
            land = mix(PAPER, INK, 0.18 + 0.06 * idx)
            draw.rectangle((x, y, x + 82, y + 94), outline=rgba(INK, alpha * 0.48), width=3)
            draw.rectangle((x + 5, y + 5, x + 77, y + 47), fill=rgba(sky, alpha * 0.82))
            draw.rectangle((x + 5, y + 47, x + 77, y + 89), fill=rgba(land, alpha * 0.56))
        points = [(1450, 612), (1512, 572), (1582, 650), (1640, 596)]
        draw.line(points, fill=rgba(accent, alpha * 0.8), width=5, joint="curve")
        for point in points:
            draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=rgba(INK, alpha * 0.75))

    elif kind == "painting":
        for idx, color in enumerate([(96, 69, 125), (180, 130, 45), (35, 106, 93), (155, 63, 46)]):
            x = 1450 + (idx % 2) * 104
            y = 190 + (idx // 2) * 118
            draw.rectangle((x, y, x + 86, y + 86), fill=rgba(color, alpha * 0.72))
            draw.rectangle((x + 11, y + 11, x + 66, y + 18 + int(44 * abs(math.sin(t + idx)))), fill=rgba(PAPER_WARM, alpha * 0.33))
        draw.arc((1452, 585, 1648, 781), 180, 360, fill=rgba(INK, alpha * 0.55), width=7)
        needle = -math.pi + math.pi * (0.2 + 0.6 * phase)
        draw.line((1550, 683, 1550 + 82 * math.cos(needle), 683 + 82 * math.sin(needle)), fill=rgba(accent, alpha * 0.86), width=7)
        draw.text((1460, 733), "CONSEQUENCE", font=FONTS["micro"], fill=rgba(INK, alpha * 0.55))

    elif kind == "final":
        names = ["product", "spirits", "photo", "travel", "paint"]
        colors = [(35, 106, 93), (180, 130, 45), (40, 95, 135), (155, 63, 46), (96, 69, 125)]
        for idx, (name, color) in enumerate(zip(names, colors)):
            y = 188 + idx * 95
            draw.text((1450, y + 11), name.upper(), font=FONTS["micro"], fill=rgba(INK, alpha * 0.68))
            draw.rectangle((1584, y, 1645, y + 48), fill=rgba(color, alpha * (0.45 + idx * 0.08)))

    x = int((-470 if int(scene["start"] / 5) % 2 == 0 else 1970) + (720 if int(scene["start"] / 5) % 2 == 0 else -760) * phase)
    draw.rectangle((x, 910, x + 740, 958), fill=rgba(accent, alpha * 0.72))


def draw_text_block(draw: ImageDraw.ImageDraw, t: float, scene: dict) -> None:
    start = scene["start"]
    end = scene["end"]
    alpha = fade_alpha(t, start, end)
    p = clamp((t - start) / (end - start))
    drift = int(34 * (ease(p) - 0.5))
    x = 118 + drift

    draw.text((118, 102), scene["label"], font=FONTS["label"], fill=rgba(MUTED, alpha * 0.95))

    title_font = FONTS["title_small"] if scene["kind"] in {"travel", "painting", "final"} else FONTS["title"]
    wrapped_body = wrap_text(draw, scene["body"], FONTS["body"], 1120)
    draw.multiline_text((x, 218), scene["title"], font=title_font, spacing=8, fill=rgba(INK, alpha), anchor=None)
    draw.multiline_text((118, 672), wrapped_body, font=FONTS["body"], spacing=10, fill=rgba(SOFT_INK, alpha))
    draw.text((118, 832), scene["meta"], font=FONTS["meta"], fill=rgba(MUTED, alpha * 0.96))


def draw_grain(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for idx in range(95):
        x = int((idx * 421 + int(t * 31) * 17) % WIDTH)
        y = int((idx * 197 + int(t * 31) * 23) % HEIGHT)
        shade = 255 if idx % 2 else 0
        alpha = 7 if idx % 2 else 5
        draw.point((x, y), fill=(shade, shade, shade, alpha))


def render_frame(t: float) -> Image.Image:
    scene = active_scene(t)
    p = clamp((t - scene["start"]) / (scene["end"] - scene["start"]))
    alpha = fade_alpha(t, scene["start"], scene["end"])

    image = Image.new("RGBA", (WIDTH, HEIGHT), PAPER + (255,))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_background(draw, t, scene)
    draw_scene_motif(draw, t, scene, p, alpha)
    draw_text_block(draw, t, scene)
    draw_grain(image, t)
    return image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=40, threshold=3)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(args.fps * args.duration)

    for index in range(total_frames):
        t = index / args.fps
        frame = render_frame(t)
        frame.save(out_dir / f"frame_{index:04d}.png", optimize=False)
        if index % args.fps == 0:
            print(f"rendered {index // args.fps:02d}s / {int(args.duration)}s")

    print(f"rendered {total_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
