"""Tests for HTML journal generation."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dmn import html_journal, journal


class HtmlJournalTests(unittest.TestCase):
    def test_build_brief_with_image_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            journal_dir = cwd / "journal"
            artifact = cwd / "data" / "artifacts" / "img_test.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"\x89PNG\r\n")

            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                md_path = journal.write_brief(
                    seed="html preview test",
                    body="## body\n\nSome text.",
                    dopamine={"total": 0.72, "alignment": 0.5, "novelty": 0.8},
                    seed_source="test",
                    tools=["wikipedia"],
                    journal_dir=journal_dir,
                    activity="image_riff",
                    artifacts=[
                        {
                            "modality": "image",
                            "bytes_path": "data/artifacts/img_test.png",
                        }
                    ],
                )
                html_path = html_journal.write_brief_html(md_path, journal_dir)
                html_text = html_path.read_text()
                self.assertIn("../data/artifacts/img_test.png", html_text)
                self.assertIn("0.720", html_text)
                self.assertIn("image_riff", html_text)
            finally:
                os.chdir(old_cwd)

    def test_index_and_today_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal_dir = Path(tmp) / "journal"
            journal_dir.mkdir()
            entries = [
                {
                    "id": 1,
                    "seed": "first question",
                    "path": str(journal_dir / "2026-01-01-first.md"),
                    "dopamine_total": 0.9,
                    "activity": "research",
                    "tools": [],
                    "created_at": 9999999999.0,
                    "mutation": "root",
                    "depth": 0,
                },
                {
                    "id": 2,
                    "seed": "second question",
                    "path": str(journal_dir / "2026-01-01-second.md"),
                    "dopamine_total": 0.4,
                    "activity": "music_riff",
                    "tools": [],
                    "created_at": 9999999999.0,
                },
            ]
            for e in entries:
                Path(e["path"]).write_text(
                    f"---\nseed: {e['seed']}\ndopamine: {{\"total\": {e['dopamine_total']}}}\n---\n\n# Brief\n"
                )
            index = html_journal.write_index_html(entries, journal_dir)
            today = html_journal.write_today_html(entries, journal_dir)
            index_text = index.read_text()
            self.assertIn("first question", index_text)
            self.assertIn("0.900", index_text)
            self.assertTrue(today.exists())


if __name__ == "__main__":
    unittest.main()
