# Vivian Artifacts

Generated local artifacts based on Vivian's imported default-mode-network profile.

## Creative Artifacts

- `the-product-sommelier.html` - creative zine with invented interface drinks, field notes, object cards, and a prompt machine.
- `the-product-sommelier-prompt.md` - source creative brief used to make the Product Sommelier artifact.
- `the-product-sommelier-video.mp4` - 30-second motion artifact based on the local profile themes.
- `vivian-website-trailer.mp4` - 47-second trailer cut from public images on Vivian's websites.
- `vivian-curiosity-atlas.html` - interactive atlas of the profile's source themes and evidence.
- `vivian-dmn-first-artifact.html` - rendered version of the first taste-map artifact.
- `vivian-dmn-first-artifact.md` - Markdown source for the first taste-map artifact.

## Previews

- `previews/the-product-sommelier-desktop.png`
- `previews/the-product-sommelier-mobile.png`
- `previews/the-product-sommelier-video-preview.png`
- `previews/vivian-website-trailer-preview.png`
- `previews/vivian-curiosity-atlas-desktop.png`
- `previews/vivian-curiosity-atlas-mobile.png`

## Video Source

- `video-source/render_product_sommelier.py` - Pillow frame renderer for the motion typography.
- `video-source/render-product-sommelier.sh` - ffmpeg wrapper that renders frames, encodes MP4, and extracts the preview still.
- `trailer-source/fetch_website_images.py` - public website image extractor/downloader.
- `trailer-source/make_contact_sheet.py` - contact-sheet helper for selecting source images.
- `trailer-source/selected-site-images/` - curated public website images used in the trailer.
- `trailer-source/render_website_trailer.py` - Pillow frame renderer for the website-image trailer.
- `trailer-source/render-website-trailer.sh` - ffmpeg wrapper that encodes the website-image trailer.

## Notes

- These are generated artifacts only.
- Raw imports, SQLite profile data, browser-history-derived records, and handoff folders are intentionally not included here.
