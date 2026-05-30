#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests


START_URLS = [
    "https://www.viviancromwell.com/",
    "https://www.viviancromwell.com/about",
    "https://www.viviancromwell.com/photography",
    "https://www.viviancromwell.com/paintings",
    "https://www.viviancromwell.com/product-design",
    "https://www.viviancromwellphotography.com/",
    "https://www.viviancromwellphotography.com/portrait",
    "https://www.viviancromwellphotography.com/maternity",
    "https://www.viviancromwellphotography.com/studio",
    "https://www.viviancromwellphotography.com/event",
    "https://www.viviancromwellphotography.com/about",
]

ALLOWED_HOSTS = {
    "www.viviancromwell.com",
    "viviancromwell.com",
    "www.viviancromwellphotography.com",
    "viviancromwellphotography.com",
    "images.squarespace-cdn.com",
    "static1.squarespace.com",
    "static.showit.co",
    "img1.wsimg.com",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MIN_BYTES = 24_000


class ImageExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value for key, value in attrs if value}
        candidates: list[str] = []
        for key in ("src", "href", "data-src", "data-image", "data-original", "content"):
            if values.get(key):
                candidates.append(values[key])
        for key in ("srcset", "data-srcset"):
            srcset = values.get(key)
            if srcset:
                candidates.extend(part.strip().split(" ")[0] for part in srcset.split(","))
        style = values.get("style", "")
        candidates.extend(match.group(1) for match in re.finditer(r"url\(['\"]?([^'\")]+)", style))

        for candidate in candidates:
            self.urls.extend(extract_image_urls(candidate, self.base_url))


def extract_image_urls(value: str, base_url: str) -> list[str]:
    value = unquote(value.strip())
    found: list[str] = []

    if value.startswith("//"):
        value = "https:" + value
    if value.startswith("/"):
        value = urljoin(base_url, value)

    if value.startswith("http"):
        parsed = urlparse(value)
        if parsed.netloc.lower() in ALLOWED_HOSTS and is_image_like(parsed.path):
            found.append(value)
        query_url = parse_qs(parsed.query).get("url", [""])[0]
        if query_url:
            found.extend(extract_image_urls(query_url, base_url))

    for match in re.finditer(r"https?://[^'\"\s<>]+", value):
        url = match.group(0).rstrip("),;")
        parsed = urlparse(url)
        if parsed.netloc.lower() in ALLOWED_HOSTS and is_image_like(parsed.path):
            found.append(url)

    return found


def is_image_like(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(IMAGE_EXTENSIONS) or "/content/" in lower or "format=" in lower


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    keep: list[tuple[str, str]] = []
    for key in ("format", "width", "height", "quality", "name"):
        for value in query.get(key, []):
            keep.append((key, value))
    query_text = "&".join(f"{key}={value}" for key, value in keep)
    return parsed._replace(query=query_text, fragment="").geturl()


def filename_for(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    stem = Path(unquote(parsed.path)).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "site-image"
    suffix = Path(stem).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg" if "jpeg" in content_type or "jpg" in content_type else ".png"
        stem = Path(stem).stem + suffix
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{Path(stem).stem}-{digest}{suffix}"


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def download(session: requests.Session, url: str, out_dir: Path) -> Path | None:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type and len(response.content) < MIN_BYTES:
        return None
    if len(response.content) < MIN_BYTES:
        return None

    out_path = out_dir / filename_for(url, content_type)
    out_path.write_bytes(response.content)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="vivian/trailer-source/site-images")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "DMN website trailer builder"})

    image_urls: list[str] = []
    for url in START_URLS:
        try:
            html = fetch_html(session, url)
        except Exception as exc:
            print(f"skip page {url}: {exc}")
            continue
        extractor = ImageExtractor(url)
        extractor.feed(html)
        for image_url in extractor.urls:
            clean = clean_url(image_url)
            if clean not in image_urls:
                image_urls.append(clean)
        print(f"found {len(extractor.urls):03d} image references on {url}")

    saved: list[Path] = []
    for image_url in image_urls[: args.limit]:
        try:
            path = download(session, image_url, out_dir)
        except Exception as exc:
            print(f"skip image {image_url}: {exc}")
            continue
        if path:
            saved.append(path)
            print(f"saved {path}")

    manifest = out_dir.parent / "site-images-manifest.txt"
    manifest.write_text("\n".join(str(path) for path in saved) + "\n", encoding="utf-8")
    print(f"wrote {manifest} ({len(saved)} images)")


if __name__ == "__main__":
    main()
