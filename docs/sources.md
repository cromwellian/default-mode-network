# Where your taste lives

A taste profile is only as good as what feeds it. The principle:

> **Taste is what you choose to consume in your free time** — watch, read,
> listen, browse, save. Not what you're obligated to do (email, calendar),
> and not what you produce (your own site, your own docs).

## Sources, ranked by signal

| signal | source | how to get it | import |
| ------ | ------ | ------------- | ------ |
| ★★★ | YouTube watch history | [takeout.google.com](https://takeout.google.com) → **Deselect all → tick YouTube** → export arrives in minutes | `--import youtube --takeout-dir …` |
| ★★★ | Saved / read-later links (Readwise, Pocket, GoodLinks, Matter…) | the app's export (CSV directly, or JSON → convert, below) | `--csv file.csv` |
| ★★★ | Deep Chrome history (synced, 18 months+) | same Takeout → also tick **Chrome** | `--import chrome --takeout-dir …` |
| ★★ | Local browser history (~last 90 days) | nothing to export — read locally (Chrome/Arc/Brave/Edge/Firefox/Safari) | `--import browser` |
| ★★ | Spotify liked songs / playlists | [exportify.net](https://exportify.net) → CSV in minutes (the official Spotify export takes days) | `--csv liked_songs.csv` |
| ★★ | Goodreads library | Goodreads → My Books → Import/Export → CSV | `--csv goodreads.csv` |
| ★★ | Twitter/X likes | Settings → Download an archive of your data | `--import twitter --twitter-dir …` |
| ★ | Gmail sent-mail subjects | same Takeout → tick Mail | `--import gmail --takeout-dir …` — logistics-heavy; expect dilution |
| ✗ | Calendar | — | not supported on purpose: obligations, not curiosity |

Everything combines in one run, and later runs **append** (re-clustering over the
union), so you can add sources as you export them.

## The generic CSV door

`--csv` (alias `--readwise-csv`) accepts any CSV with a `Title` / `title` /
`Highlight` / `text` / `Note` column. Anything that exports JSON can be
converted — one title per row is all DMN needs. If you're using an AI agent,
just hand it the export and say "convert this to a title CSV and import it."

## Safari note

macOS protects Safari's history (TCC): grant your terminal **Full Disk Access**
(System Settings → Privacy & Security) or skip Safari. Chromium browsers read
without any prompt; the importer takes a read-only temp copy either way and
processes everything on your machine — the filter report shows exactly what was
kept and dropped.
