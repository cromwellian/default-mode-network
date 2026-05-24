"""Focused tests for v0.5 wander controls and activities."""
from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dmn import journal, store
from dmn.activities import ActivityContext, modality_weight_factor
from dmn.activities.web_app_sketch import WebAppSketchActivity
from dmn.seeds import Seed, _moderate_distance_pair
from dmn.tree import Frontier
from wander import _prune_frontier_to_beam


class StubLLM:
    name = "stub"

    def complete(self, *args, **kwargs):
        raise RuntimeError("dry-run path should not call llm")


def _ctx(tmp: Path) -> ActivityContext:
    return ActivityContext(
        llm=StubLLM(),
        embed_fn=lambda texts: np.zeros((len(texts), 4), dtype=np.float32),
        clusters=[],
        centroids=[],
        recent_embs=[],
        rng=random.Random(0),
        dry_run=True,
        artifact_root=tmp / "data" / "artifacts",
    )


class V05Tests(unittest.TestCase):
    def test_cross_pollination_uses_moderate_band(self) -> None:
        clusters = [
            {"id": 0, "label": "a", "centroid": np.array([1.0, 0.0])},
            {"id": 1, "label": "b", "centroid": np.array([0.98, 0.17])},
            {"id": 2, "label": "c", "centroid": np.array([0.50, 0.86])},
            {"id": 3, "label": "d", "centroid": np.array([-0.50, 0.86])},
            {"id": 4, "label": "e", "centroid": np.array([-1.0, 0.0])},
        ]
        rng = random.Random(3)
        a, b = _moderate_distance_pair(clusters, rng)
        pair_ids = {a["id"], b["id"]}
        self.assertNotEqual(pair_ids, {0, 1})  # closest
        self.assertNotEqual(pair_ids, {0, 4})  # farthest

    def test_modality_gating_downweights_unrelated_seed(self) -> None:
        finance = "duration hedging and convexity in fixed income portfolios"
        self.assertLess(modality_weight_factor("image_riff", seed_text=finance), 0.2)
        self.assertLess(modality_weight_factor("music_riff", seed_text=finance), 0.2)
        self.assertEqual(
            modality_weight_factor("image_riff", seed_text="generative art diagram"),
            1.0,
        )
        self.assertEqual(
            modality_weight_factor("music_riff", seed_text="polyrhythm music tool"),
            1.0,
        )

    def test_web_app_sketch_writes_index_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                activity = WebAppSketchActivity()
                result = activity.run(Seed("wave interference", "manual"), _ctx(cwd))
                artifact_path = result.artifacts[0].bytes_path
                self.assertIsNotNone(artifact_path)
                assert artifact_path is not None
                self.assertTrue(artifact_path.exists())
                self.assertTrue((artifact_path.parent / "manifest.json").exists())
                brief = journal.write_brief(
                    seed="wave interference",
                    body=result.body_md,
                    dopamine={"total": 0.5},
                    seed_source="manual",
                    tools=[],
                    artifacts=[
                        {
                            "modality": a.modality,
                            "bytes_path": str(a.bytes_path) if a.bytes_path else None,
                            "url": a.url,
                        }
                        for a in result.artifacts
                    ],
                )
                self.assertIn("<iframe", brief.read_text())
            finally:
                os.chdir(old_cwd)

    def test_beam_pruning_marks_extra_open_leaves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(Path(tmp) / "dmn.sqlite")
            frontier = Frontier()
            ids: list[int] = []
            for score in [0.9, 0.4, 0.8, 0.2]:
                jid = store.add_journal(
                    conn,
                    seed=f"seed {score}",
                    seed_source="test",
                    tools=[],
                    dopamine={"total": score},
                    path=f"journal/{score}.md",
                    run_id="run-a",
                    status="open",
                )
                ids.append(jid)
                frontier.push(jid, score)

            pruned = _prune_frontier_to_beam(conn, frontier, "run-a", beam_width=2)
            rows = {row["id"]: row for row in store.list_journal_by_run(conn, "run-a")}
            self.assertEqual(pruned, 2)
            self.assertEqual(frontier.size(), 2)
            self.assertEqual(rows[ids[1]]["status"], "pruned")
            self.assertEqual(rows[ids[3]]["status"], "pruned")
            self.assertEqual(rows[ids[0]]["status"], "open")
            self.assertEqual(rows[ids[2]]["status"], "open")


if __name__ == "__main__":
    unittest.main()
