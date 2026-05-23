"""Tests for journal-relative artifact hrefs."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dmn.journal import write_brief
from dmn.paths import artifact_href, artifact_href_for_journal


class ArtifactHrefTests(unittest.TestCase):
    def test_journal_relative_image_path(self) -> None:
        href = artifact_href_for_journal("journal", "data/artifacts/img_test.png")
        self.assertEqual(href, "../data/artifacts/img_test.png")

    def test_brief_path_nested_journal(self) -> None:
        href = artifact_href(
            Path("journal/2026-05-23-foo.md"),
            "data/artifacts/img_test.png",
        )
        self.assertEqual(href, "../data/artifacts/img_test.png")

    def test_url_unchanged(self) -> None:
        url = "https://example.com/audio.mp3"
        self.assertEqual(
            artifact_href(Path("journal/foo.md"), url),
            url,
        )

    def test_write_brief_embeds_journal_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            journal_dir = cwd / "journal"
            artifact = cwd / "data" / "artifacts" / "img_test.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"\x89PNG\r\n")

            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                brief_path = write_brief(
                    seed="preview test",
                    body="## body",
                    dopamine={"total": 0.5},
                    seed_source="test",
                    tools=[],
                    journal_dir=journal_dir,
                    artifacts=[
                        {
                            "modality": "image",
                            "bytes_path": "data/artifacts/img_test.png",
                            "url": None,
                        }
                    ],
                )
                text = brief_path.read_text()
                self.assertIn("![generated](../data/artifacts/img_test.png)", text)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
