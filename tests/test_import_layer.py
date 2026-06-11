"""Workspace-noise category, takeout-chrome importer, profile-quality readout (#36/#38/#27)."""
from __future__ import annotations

import json
from pathlib import Path

from dmn.importers import browser as browser_imp
from dmn.importers._classify import is_workspace_tool
from prepare import profile_quality_report


def test_workspace_tools_classified():
    assert is_workspace_tool("https://docs.google.com/document/d/abc/edit")
    assert is_workspace_tool("https://mycorp.atlassian.net/browse/ENG-123")
    assert is_workspace_tool("https://calendar.google.com/calendar/u/0/r/week")
    assert not is_workspace_tool("https://en.wikipedia.org/wiki/Polyrhythm")
    assert not is_workspace_tool("https://substack.com/some-essay")


def test_clean_with_stats_drops_work_tools_by_default():
    rows = [
        {
            "title": "Q3 Budget Planning",
            "url": "https://docs.google.com/spreadsheets/d/xyz/edit",
            "browser": "chrome",
            "visit_count": 40,
            "weight": 3.7,
        },
        {
            "title": "The Strange Loop of Self-Reference in Music",
            "url": "https://example-essays.com/strange-loop-music",
            "browser": "chrome",
            "visit_count": 2,
            "weight": 1.1,
        },
    ]
    kept, drops, _gate = browser_imp._clean_with_stats(rows, keep_services=False)
    assert drops.get("work_tool", 0) == 1
    assert len(kept) == 1
    kept_all, _, _ = browser_imp._clean_with_stats(rows, keep_services=True)
    assert len(kept_all) == 2  # escape hatch keeps work tools


def test_takeout_chrome_importer(tmp_path: Path):
    chrome_dir = tmp_path / "Takeout" / "Chrome"
    chrome_dir.mkdir(parents=True)
    entries = [
        {"title": "Kintsugi and the Aesthetics of Repair", "url": "https://essays.example.com/kintsugi", "time_usec": 1_700_000_000_000_000},
        {"title": "Kintsugi and the Aesthetics of Repair", "url": "https://essays.example.com/kintsugi", "time_usec": 1_700_100_000_000_000},
        {"title": "Sprint Planning", "url": "https://mycorp.atlassian.net/browse/ENG-9", "time_usec": 1_700_000_000_000_000},
    ]
    (chrome_dir / "BrowserHistory.json").write_text(json.dumps({"Browser History": entries}))
    out = browser_imp.import_takeout_history(tmp_path / "Takeout")
    texts = [o["text"] for o in out]
    assert any("Kintsugi" in t for t in texts)
    assert not any("Sprint Planning" in t for t in texts)  # work tool filtered
    assert all(o["source"] == "browser:takeout-chrome" for o in out)


def test_takeout_chrome_missing_file(tmp_path: Path):
    assert browser_imp.import_takeout_history(tmp_path) == []


def test_profile_quality_report_flags_dominance_and_depth():
    skewed = [{"source": "website"}] * 86 + [{"source": "browser:chrome"}] * 14
    lines = profile_quality_report(skewed)
    assert any("86%" in l and "website" in l for l in lines)
    thin = [{"source": "manual"}] * 8
    lines = profile_quality_report(thin)
    assert any("small profile" in l for l in lines)
    healthy = [{"source": "manual"}] * 20 + [{"source": "browser:chrome"}] * 20
    assert len(profile_quality_report(healthy)) == 1  # breakdown only, no warnings
