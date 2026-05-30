#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


WIDTH = 1920
HEIGHT = 1080

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "vivian/trailer-source/selected-site-images"

SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"
SERIF_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
SANS_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


@dataclass(frozen=True)
class Shot:
    image: str
    duration: float
    text: str = ""
    subtext: str = ""
    text_at: str = "left"
    start_zoom: float = 1.02
    end_zoom: float = 1.14
    pan_x: float = 0.5
    pan_y: float = 0.5
    pan_dx: float = 0.0
    pan_dy: float = 0.0
    grade: str = "warm"


SHOTS = [
    Shot("vivian-portrait.B6EB_6BO_Z1HJeO3-83d6f88c.webp", 3.2, "THE ARCHIVE\nLEARNED TO TASTE", "cut from Vivian's public visual archive", "left", 1.02, 1.10, 0.48, 0.50, 0.02, 0.0, "mono"),
    Shot("001.CnSFHAXn_Z3eYMJ-45486034.webp", 2.7, "It began\nwith looking.", "", "right", 1.03, 1.16, 0.50, 0.48, 0.0, 0.04, "mono"),
    Shot("004.C0AusSYH_1JvbW8-a8275e24.webp", 2.2, "", "", "left", 1.02, 1.18, 0.48, 0.50, 0.04, 0.0, "mono"),
    Shot("005.Blongk-3_sJbOw-b5a2b7cd.webp", 2.4, "Every city\nleft a trace.", "", "left", 1.00, 1.12, 0.44, 0.55, 0.07, -0.02, "warm"),
    Shot("007.Dv3oWx-0_tGJRi-4c47fcb0.webp", 2.1, "", "", "left", 1.00, 1.13, 0.48, 0.50, 0.02, 0.0, "warm"),
    Shot("009.PvXbLg9J_Z1OeNDq-d926e22a.webp", 2.4, "Rooms remembered\nwhat people almost said.", "", "left", 1.04, 1.19, 0.52, 0.52, -0.04, 0.0, "warm"),
    Shot("010.B85kii8v_ZtI0pl-29575106.webp", 2.0, "", "", "left", 1.04, 1.17, 0.48, 0.48, 0.03, 0.01, "warm"),
    Shot("015.BCanGhfh_1JLSvP-99fd27c7.webp", 2.2, "Some memories\ncame back as heat.", "", "right", 1.02, 1.18, 0.50, 0.50, 0.0, -0.04, "warm"),
    Shot("016.NfWPTEir_1rWgzb-1a2f7adb.webp", 2.1, "", "", "left", 1.01, 1.13, 0.50, 0.50, -0.02, 0.0, "warm"),
    Shot("014.DB8KBn30_MLtnc-ea8e55f9.webp", 2.3, "Some as distance.", "", "left", 1.03, 1.18, 0.50, 0.52, 0.0, -0.05, "cool"),
    Shot("001.BJ8P5xPR_aKmiW-3f89ee06.webp", 2.5, "Then the machine\nfound a face in the dark.", "", "right", 1.03, 1.15, 0.50, 0.47, 0.0, 0.03, "mono"),
    Shot("004.Dzc_J7_P_ZoBqjI-14c60c4b.webp", 2.2, "", "", "left", 1.02, 1.15, 0.51, 0.48, -0.01, 0.02, "mono"),
    Shot("blue-vase-and-apples.BblXA4b7_1XJh30-e2e0c494.webp", 2.5, "It mistook still life\nfor evidence.", "", "left", 1.04, 1.22, 0.52, 0.50, -0.05, 0.0, "paint"),
    Shot("autumn-at-pellehaut.JiS5-aeR_2cCb0J-3c6e06fd.webp", 2.2, "", "", "left", 1.02, 1.17, 0.50, 0.50, 0.03, -0.02, "paint"),
    Shot("golden-hour-ed-r-levin.MB-C-687_JRJam-d33c5873.webp", 2.2, "Paint became\na weather report.", "", "right", 1.02, 1.18, 0.50, 0.48, 0.0, 0.04, "paint"),
    Shot("snowed-in-breckenridge.DX0tL8GD_Z21uBVb-754d6723.webp", 2.0, "", "", "left", 1.03, 1.16, 0.50, 0.50, -0.03, 0.0, "paint"),
    Shot("002.C8F1kg_5_Z5KV1D-9c18d673.webp", 1.4, "", "", "left", 1.05, 1.20, 0.50, 0.50, 0.02, 0.0, "warm"),
    Shot("006.DW0Mrd0i_Z1VMfu4-74c3877b.webp", 1.2, "", "", "left", 1.05, 1.22, 0.52, 0.50, -0.02, 0.0, "mono"),
    Shot("011.C6DFkdt0_stxbh-f7257845.webp", 1.2, "", "", "left", 1.03, 1.17, 0.50, 0.50, 0.03, -0.02, "mono"),
    Shot("001.BJ8P5xPR_aKmiW-3f89ee06.webp", 1.2, "", "", "left", 1.07, 1.28, 0.50, 0.48, 0.0, 0.02, "mono"),
    Shot("vivian-portrait.B6EB_6BO_Z1HJeO3-83d6f88c.webp", 4.0, "She did not ask it\nto remember.", "She asked it to taste.", "left", 1.02, 1.10, 0.48, 0.50, 0.0, 0.0, "mono"),
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


FONTS = {
    "title": font(SERIF_BOLD, 104),
    "body": font(SERIF, 72),
    "small": font(SANS_BOLD, 28),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    value = clamp(value)
    return value * value * (3 - 2 * value)


def load_images() -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for shot in SHOTS:
        path = IMAGE_DIR / shot.image
        with Image.open(path) as source:
            images[shot.image] = source.convert("RGB")
    return images


def grade_image(image: Image.Image, grade: str) -> Image.Image:
    if grade == "mono":
        gray = image.convert("L").convert("RGB")
        image = Image.blend(gray, image, 0.15)
        image = ImageEnhance.Contrast(image).enhance(1.28)
        image = ImageEnhance.Brightness(image).enhance(0.88)
        return image
    if grade == "cool":
        r, g, b = image.split()
        image = Image.merge("RGB", (r.point(lambda v: v * 0.92), g, b.point(lambda v: min(255, v * 1.08))))
        return ImageEnhance.Contrast(image).enhance(1.12)
    if grade == "paint":
        image = ImageEnhance.Color(image).enhance(1.08)
        image = ImageEnhance.Contrast(image).enhance(1.08)
        return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=30, threshold=4))
    image = ImageEnhance.Color(image).enhance(1.06)
    image = ImageEnhance.Contrast(image).enhance(1.10)
    return image


def cover_frame(image: Image.Image, shot: Shot, progress: float) -> Image.Image:
    progress = ease(progress)
    zoom = shot.start_zoom + (shot.end_zoom - shot.start_zoom) * progress
    center_x = shot.pan_x + shot.pan_dx * progress
    center_y = shot.pan_y + shot.pan_dy * progress
    base = max(WIDTH / image.width, HEIGHT / image.height) * zoom
    scaled_w = max(WIDTH, int(image.width * base))
    scaled_h = max(HEIGHT, int(image.height * base))
    resized = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    max_x = scaled_w - WIDTH
    max_y = scaled_h - HEIGHT
    left = int(clamp(center_x, 0.0, 1.0) * max_x)
    top = int(clamp(center_y, 0.0, 1.0) * max_y)
    return resized.crop((left, top, left + WIDTH, top + HEIGHT))


def cumulative_times() -> list[float]:
    times = [0.0]
    total = 0.0
    for shot in SHOTS:
        total += shot.duration
        times.append(total)
    return times


def shot_at(t: float, times: list[float]) -> tuple[int, float]:
    for index in range(len(SHOTS)):
        if times[index] <= t < times[index + 1]:
            return index, t - times[index]
    return len(SHOTS) - 1, SHOTS[-1].duration


def draw_text(image: Image.Image, shot: Shot, local_t: float) -> None:
    if not shot.text:
        return
    fade = min(clamp(local_t / 0.55), clamp((shot.duration - local_t) / 0.55))
    if fade <= 0:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    x = 132 if shot.text_at == "left" else 1000
    y = 650 if "\n" in shot.text else 710
    if shot.image == "vivian-portrait.B6EB_6BO_Z1HJeO3-83d6f88c.webp" and shot.text.startswith("THE"):
        x = 112
        y = 104
    title_font = FONTS["title"] if shot.text.startswith("THE") else FONTS["body"]
    alpha = int(255 * fade)
    shadow = (0, 0, 0, int(135 * fade))
    fill = (250, 242, 226, alpha)

    for dx, dy in ((4, 4), (2, 2)):
        draw.multiline_text((x + dx, y + dy), shot.text, font=title_font, spacing=9, fill=shadow)
    draw.multiline_text((x, y), shot.text, font=title_font, spacing=9, fill=fill)

    if shot.subtext:
        sy = y + 250 if shot.text.startswith("THE") else y + 190
        draw.text((x + 2, sy + 2), shot.subtext.upper(), font=FONTS["small"], fill=shadow)
        draw.text((x, sy), shot.subtext.upper(), font=FONTS["small"], fill=(232, 218, 190, alpha))


def overlays(image: Image.Image, frame_index: int, t: float, duration: float) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    draw.rectangle((0, 0, WIDTH, 92), fill=(0, 0, 0, 72))
    draw.rectangle((0, HEIGHT - 94, WIDTH, HEIGHT), fill=(0, 0, 0, 82))
    draw.rectangle((0, 0, WIDTH, HEIGHT), outline=(255, 246, 225, 36), width=2)

    fade_in = clamp(t / 1.2)
    fade_out = clamp((duration - t) / 1.4)
    fade = min(fade_in, fade_out)
    if fade < 1:
        draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, int(255 * (1 - fade))))

    rng = random.Random(frame_index)
    for _ in range(85):
        x = rng.randrange(WIDTH)
        y = rng.randrange(HEIGHT)
        a = rng.randrange(8, 22)
        color = 255 if rng.random() > 0.55 else 0
        draw.point((x, y), fill=(color, color, color, a))

    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def render_frame(t: float, frame_index: int, images: dict[str, Image.Image], times: list[float], duration: float) -> Image.Image:
    shot_index, local_t = shot_at(t, times)
    shot = SHOTS[shot_index]
    progress = local_t / shot.duration
    current = grade_image(cover_frame(images[shot.image], shot, progress), shot.grade)

    dissolve = 0.42
    if local_t > shot.duration - dissolve and shot_index < len(SHOTS) - 1:
        next_shot = SHOTS[shot_index + 1]
        next_progress = (local_t - (shot.duration - dissolve)) / dissolve * 0.12
        next_frame = grade_image(cover_frame(images[next_shot.image], next_shot, next_progress), next_shot.grade)
        amount = ease((local_t - (shot.duration - dissolve)) / dissolve)
        current = Image.blend(current, next_frame, amount)

    draw_text(current, shot, local_t)
    return overlays(current, frame_index, t, duration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = cumulative_times()
    duration = times[-1]
    total_frames = int(math.ceil(duration * args.fps))
    images = load_images()

    for frame_index in range(total_frames):
        t = frame_index / args.fps
        frame = render_frame(t, frame_index, images, times, duration)
        frame.save(out_dir / f"frame_{frame_index:04d}.jpg", quality=91)
        if frame_index % args.fps == 0:
            print(f"rendered {frame_index // args.fps:02d}s / {int(duration)}s")

    print(f"rendered {total_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
