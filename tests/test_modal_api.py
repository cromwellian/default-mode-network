"""Tests for wander runner argv mapping and API response serialization."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dmn.api_paths import artifact_url, validate_user_id
from dmn.api_response import build_wander_response
from dmn.wander_config import WanderConfig, WanderResult
from dmn.wander_runner import build_wander_argv


class TestApiPaths(unittest.TestCase):
    def test_validate_user_id(self) -> None:
        self.assertEqual(validate_user_id("alice"), "alice")
        self.assertEqual(validate_user_id("user_123"), "user_123")
        with self.assertRaises(ValueError):
            validate_user_id("../etc")
        with self.assertRaises(ValueError):
            validate_user_id("")

    def test_artifact_url(self) -> None:
        url = artifact_url(
            "https://example.modal.run",
            "alice",
            "abc123def456",
            "def456abc123",
            "journal/tree.html",
        )
        self.assertEqual(
            url,
            "https://example.modal.run/v1/users/alice/profiles/abc123def456/"
            "runs/def456abc123/files/journal/tree.html",
        )


class TestWanderRunner(unittest.TestCase):
    def test_build_argv_minimal(self) -> None:
        cfg = WanderConfig(iterations=1, root_count=1, max_depth=1)
        argv = build_wander_argv(cfg)
        self.assertIn("--iterations", argv)
        self.assertIn("1", argv)
        self.assertIn("--root-count", argv)
        self.assertIn("--max-depth", argv)

    def test_build_argv_flags(self) -> None:
        cfg = WanderConfig(dry_run=True, no_execute=True, seed_text="hello world")
        argv = build_wander_argv(cfg)
        self.assertIn("--dry-run", argv)
        self.assertIn("--no-execute", argv)
        self.assertIn("--seed", argv)
        self.assertIn("hello world", argv)


class TestApiResponse(unittest.TestCase):
    def test_build_wander_response_inlines_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            journal = work / "journal"
            journal.mkdir()
            brief = journal / "brief.md"
            brief.write_text("---\nseed: \"x\"\n---\n\n# Hello\n", encoding="utf-8")
            result = WanderResult(
                run_id="abc123def456",
                work_dir=str(work),
                brief_count=1,
                best_score=0.42,
                pruned_count=0,
                leaf_count=0,
                open_count=1,
                duration_seconds=12.0,
                patience_triggered=False,
                beam_pruned=0,
                briefs=[{"seed": "x", "path": "journal/brief.md"}],
                report_path="journal/brief.md",
                files={"journal/brief.md": str(brief)},
            )
            payload = build_wander_response(
                result,
                user_id="alice",
                profile_id="abc123def456",
                base_url="https://api.test",
            )
            self.assertEqual(payload["user_id"], "alice")
            self.assertEqual(payload["run_id"], "abc123def456")
            self.assertEqual(len(payload["artifacts"]), 1)
            self.assertIn("content", payload["artifacts"][0])
            self.assertIn("urls", payload)


if __name__ == "__main__":
    unittest.main()
