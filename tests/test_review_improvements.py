"""Tests for the post-v0.5 review fixes:

  - search-query derivation (conversational seed -> keyword query for search APIs)
  - seed generators populating `Seed.query`
  - the dopamine reward (base math + serendipity gating + the EPS/bonus split)
  - UCB-aware Frontier selection (explore vs. exploit)
"""
from __future__ import annotations

import random
import unittest

import numpy as np

from dmn import seeds, taste
from dmn.activities._helpers import extract_fenced, extract_html
from dmn.seeds import Seed, search_query
from dmn.tree import Frontier


class FixedRng:
    """Minimal rng stub: dopamine() calls .random() exactly once for the serendipity gate."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


class SearchQueryTests(unittest.TestCase):
    def test_strips_fresh_angle_scaffolding(self) -> None:
        self.assertEqual(
            search_query("What's a fresh angle on contrarian nutrition science?"),
            "contrarian nutrition science",
        )

    def test_strips_deeper_story_prefix(self) -> None:
        self.assertEqual(
            search_query("What is the deeper story behind: Foo Bar"),
            "Foo Bar",
        )

    def test_strips_leading_article(self) -> None:
        self.assertEqual(
            search_query("The economics of attention"),
            "economics of attention",
        )

    def test_falls_back_to_subtopic_when_empty(self) -> None:
        # After stripping the interrogative the query is too short, so the subtopic wins.
        self.assertEqual(search_query("Why?", subtopic="genomics"), "genomics")

    def test_no_leading_what_in_output(self) -> None:
        q = search_query("What's a fresh angle on RLHF preference datasets?")
        self.assertFalse(q.lower().startswith("what"))


class SeedQueryPopulationTests(unittest.TestCase):
    def _clusters(self) -> list[dict]:
        return [
            {
                "id": 0,
                "label": "fermentation",
                "centroid": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "is_noise": False,
                "n_members": 10,
                "meta": {"theme": "fermentation", "subtopics": ["lacto-fermentation"]},
            },
            {
                "id": 1,
                "label": "lisp",
                "centroid": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "is_noise": False,
                "n_members": 10,
                "meta": {"theme": "lisp macros", "subtopics": ["hygienic macros"]},
            },
        ]

    def test_cold_start_sets_clean_query(self) -> None:
        seed = seeds.cold_start(self._clusters(), random.Random(0))
        self.assertTrue(seed.query)
        self.assertFalse(seed.query.lower().startswith("what"))
        # cold_start with subtopics sets query == subtopic
        if seed.subtopic:
            self.assertEqual(seed.query, seed.subtopic)


class DopamineTests(unittest.TestCase):
    def _orthogonal_setup(self):
        centroids = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
        item = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # alignment 0
        recent = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]  # novelty 1.0
        return item, centroids, recent

    def test_base_total_without_serendipity(self) -> None:
        item, centroids, recent = self._orthogonal_setup()
        d = taste.dopamine(item, centroids, recent, FixedRng(0.99), fulfillment=1.0)
        self.assertEqual(d["serendipity"], 0.0)
        expected = taste.BETA * 1.0 + taste.DELTA * 1.0  # alignment & surprise are 0
        self.assertAlmostEqual(d["total"], round(expected, 4), places=4)

    def test_serendipity_uses_bonus_weight_not_eps(self) -> None:
        item, centroids, recent = self._orthogonal_setup()
        fired = taste.dopamine(item, centroids, recent, FixedRng(0.0), fulfillment=1.0)
        not_fired = taste.dopamine(item, centroids, recent, FixedRng(0.99), fulfillment=1.0)
        self.assertEqual(fired["serendipity"], 1.0)
        # Regression guard for the EPS double-duty bug: serendipity must contribute
        # SERENDIPITY_BONUS (0.10), not EPS (0.05), to the total.
        self.assertAlmostEqual(
            fired["total"] - not_fired["total"], taste.SERENDIPITY_BONUS, places=4
        )

    def test_serendipity_gated_by_fulfillment_floor(self) -> None:
        item, centroids, recent = self._orthogonal_setup()
        # Below the floor, even a "firing" rng roll can't trigger the serendipity jackpot.
        d = taste.dopamine(item, centroids, recent, FixedRng(0.0), fulfillment=0.1)
        self.assertEqual(d["serendipity"], 0.0)


class FrontierUCBTests(unittest.TestCase):
    def test_greedy_picks_highest_score(self) -> None:
        f = Frontier()
        f.push(1, 0.50, payload={"depth": 5})
        f.push(2, 0.48, payload={"depth": 0})
        best = f.pop_best()  # explore_c defaults to 0 -> pure best-first
        self.assertEqual(best["id"], 1)

    def test_ucb_favors_shallow_node(self) -> None:
        f = Frontier()
        f.push(1, 0.50, payload={"depth": 5})  # higher score, deep/committed branch
        f.push(2, 0.48, payload={"depth": 0})  # slightly lower, shallow/unexplored
        best = f.pop_best(total_iters=10, explore_c=0.1)
        self.assertEqual(best["id"], 2)
        self.assertEqual(best["depth"], 0)

    def test_ucb_pop_removes_node(self) -> None:
        f = Frontier()
        f.push(1, 0.50, payload={"depth": 0})
        f.pop_best(total_iters=10, explore_c=0.1)
        self.assertEqual(f.size(), 0)
        self.assertIsNone(f.pop_best(total_iters=10, explore_c=0.1))


class GroundingTests(unittest.TestCase):
    """Creation activities must consume external references, not hallucinate from the seed."""

    def _ctx(self, dry_run=False):
        from dmn.activities import ActivityContext

        return ActivityContext(
            llm=object(),
            embed_fn=lambda t: np.zeros((len(t), 4), dtype=np.float32),
            clusters=[], centroids=[], recent_embs=[],
            rng=random.Random(0), dry_run=dry_run,
        )

    def test_gather_uses_clean_query_and_caches(self):
        import dmn.grounding as g
        from dmn.tools import ResearchItem

        calls = {"queries": []}

        def fake_exec(tools, query, verbose=False, log=None):
            calls["queries"].append(query)
            return [ResearchItem(title="Bandit Algorithms", summary="UCB1 etc.",
                                 url="http://x", source="arxiv")]

        g.execute_tools = fake_exec
        g.available_tools_for = lambda dry: ["arxiv", "wikipedia"]
        ctx = self._ctx()
        seed = Seed(text="What's a fresh angle on multi-armed bandits?",
                    source="cold", query="multi-armed bandits")
        out1 = g.gather_grounding(seed, ctx)
        out2 = g.gather_grounding(seed, ctx)  # cached: no second search
        self.assertEqual(len(out1), 1)
        self.assertEqual(calls["queries"], ["multi-armed bandits"])  # clean query, one call
        self.assertIs(out1, out2)

    def test_ground_disabled_returns_empty(self):
        import dmn.grounding as g
        ctx = self._ctx()
        ctx.ground = False
        out = g.gather_grounding(Seed(text="x", source="cold", query="x"), ctx)
        self.assertEqual(out, [])

    def test_block_instructs_to_build_on_refs(self):
        import dmn.grounding as g
        from dmn.tools import ResearchItem
        refs = [ResearchItem(title="Paper A", summary="s", url="http://a", source="arxiv")]
        block = g.grounding_block(refs)
        self.assertIn("Paper A", block)
        self.assertIn("build on", block.lower())

    def test_block_falls_back_when_no_refs(self):
        import dmn.grounding as g
        self.assertIn("first principles", g.grounding_block([]).lower())

    def test_fulfillment_rewards_grounding(self):
        import dmn.grounding as g
        from dmn.tools import ResearchItem
        r = ResearchItem(title="t", summary="s", url="u", source="arxiv")
        self.assertEqual(g.grounding_fulfillment([r, r]), 1.0)
        self.assertEqual(g.grounding_fulfillment([r]), 0.85)
        self.assertLess(g.grounding_fulfillment([]), 0.85)  # ungrounded penalized


class NewSourceTests(unittest.TestCase):
    """New grounding sources (github / huggingface / rss) register and behave."""

    def test_new_tools_registered(self):
        from dmn.tools import available
        for name in ("github", "huggingface", "rss"):
            self.assertIn(name, available())

    def test_grounding_prefers_new_sources(self):
        import dmn.grounding as g
        for name in ("github", "huggingface", "rss"):
            self.assertIn(name, g._PREFERRED)

    def test_rss_query_ranks_by_keyword_overlap(self):
        import dmn.tools.rss as rss
        from dmn.tools import ResearchItem
        items = [
            ResearchItem(title="Quantum error correction milestone", summary="qubits",
                         url="u1", source="rss:quanta"),
            ResearchItem(title="New LLM reasoning benchmark released", summary="llm eval",
                         url="u2", source="rss:techmeme"),
        ]
        rss._CACHE["items"] = items
        rss._CACHE["ts"] = 2**40  # force cache hit, no network
        top = rss.search("llm benchmark", max_results=1)
        self.assertEqual(len(top), 1)
        self.assertIn("LLM", top[0].title)


class StorePersistenceTests(unittest.TestCase):
    """v8: grounding refs + fulfillment must round-trip through the journal store."""

    def test_grounding_and_fulfillment_roundtrip(self):
        import tempfile
        from dmn import store
        c = store.connect(tempfile.mktemp(suffix=".sqlite"))
        jid = store.add_journal(
            c, "seed", "cold", ["wikipedia"], {"total": 0.5}, "/tmp/x.md",
            activity="code_sketch", fulfillment=1.0,
            fulfillment_breakdown={"count_score": 1.0},
            grounding=[{"title": "Bandit Book", "url": "http://x", "source": "github"}],
        )
        row = store.get_journal(c, jid)
        self.assertEqual(row["fulfillment"], 1.0)
        self.assertEqual(row["grounding"][0]["source"], "github")
        self.assertEqual(row["fulfillment_breakdown"]["count_score"], 1.0)

    def test_schema_version_and_user_rating_roundtrip(self):
        import tempfile
        from dmn import store

        c = store.connect(tempfile.mktemp(suffix=".sqlite"))
        v = c.execute("PRAGMA user_version").fetchone()[0]
        self.assertGreaterEqual(v, store.SCHEMA_VERSION)
        jid = store.add_journal(
            c, "seed", "cold", [], {"total": 0.4}, "/tmp/x.md"
        )
        store.update_journal_rating(c, jid, 4.5, "nice")
        row = store.get_journal(c, jid)
        self.assertEqual(row["user_rating"], 4.5)
        self.assertEqual(row["user_feedback"], "nice")
        self.assertIsNotNone(row["reviewed_at"])


class EvaluationTests(unittest.TestCase):
    def test_eval_summary_correlates_components(self):
        from dmn import eval as dmn_eval

        rows = [
            {"user_rating": 1, "dopamine_total": 0.1, "dopamine_breakdown": {"total": 0.1, "alignment": 0.1}},
            {"user_rating": 3, "dopamine_total": 0.5, "dopamine_breakdown": {"total": 0.5, "alignment": 0.5}},
            {"user_rating": 5, "dopamine_total": 0.9, "dopamine_breakdown": {"total": 0.9, "alignment": 0.9}},
        ]
        summary = dmn_eval.summarize(rows)
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["correlations"]["total"], 1.0)


class RetoolTests(unittest.TestCase):
    def test_research_activity_respects_forced_tools(self):
        import dmn.activities.research as research
        from dmn.activities import ActivityContext
        from dmn.tools import ResearchItem

        calls = []

        def fake_exec(tools, query, verbose=False, log=None):
            calls.append(list(tools))
            return [ResearchItem(title="Forced result", summary="Useful result", url="http://x", source=tools[0])]

        old_exec = research.execute_tools
        research.execute_tools = fake_exec

        class LLM:
            name = "stub"

            def complete(self, *args, **kwargs):
                class R:
                    text = "## What surprised me\nx\n\n## One thing you'll find delightful\nx\n\n## A rabbit hole for tomorrow\nx"

                return R()

        ctx = ActivityContext(
            llm=LLM(),
            embed_fn=lambda texts: np.ones((len(texts), 4), dtype=np.float32),
            clusters=[],
            centroids=[np.ones(4, dtype=np.float32)],
            recent_embs=[],
            rng=random.Random(0),
            dry_run=True,
            forced_tools=["hackernews"],
        )
        try:
            result = research.ResearchActivity().run(Seed("topic", "retool"), ctx)
        finally:
            research.execute_tools = old_exec
        self.assertEqual(calls[0], ["hackernews"])
        self.assertTrue(result.metadata["forced_tools"])


class PortabilityStaleTests(unittest.TestCase):
    def test_append_import_marks_profile_stale(self):
        import tempfile
        from dmn import portability, store

        c = store.connect(tempfile.mktemp(suffix=".sqlite"))
        payload = {
            "schema_version": portability.SCHEMA_VERSION,
            "embedding_model": {
                "fingerprint": portability.compute_embedding_fingerprint(),
            },
            "interests": [
                {
                    "text": "friend taste",
                    "source": "imported",
                    "weight": 1.0,
                    "embedding": [0.0] * 384,
                }
            ],
            "clusters": [],
        }
        portability.deserialize_profile(payload, c, mode="append")
        marker = store.get_meta(c, "profile_needs_recluster")
        self.assertTrue(marker["stale"])


class SeedDiversityTests(unittest.TestCase):
    """Roots should span clusters, not collapse onto the heaviest one."""

    def test_diverse_root_ordering(self):
        from wander import _diverse_root_clusters
        clusters = [
            {"id": 0, "is_noise": False, "centroid": [1, 0, 0], "n_members": 100},
            {"id": 1, "is_noise": False, "centroid": [0.99, 0.14, 0], "n_members": 80},
            {"id": 2, "is_noise": False, "centroid": [0, 1, 0], "n_members": 10},
            {"id": 3, "is_noise": False, "centroid": [0, 0, 1], "n_members": 5},
            {"id": 4, "is_noise": True, "centroid": [0.5, 0.5, 0.5], "n_members": 3},
        ]
        order = [c["id"] for c in _diverse_root_clusters(clusters, random.Random(0))]
        self.assertNotIn(4, order)  # noise excluded
        self.assertEqual(order[0], 0)  # heaviest first
        self.assertIn(order[1], (2, 3))  # then a distant cluster, not the near-duplicate (1)


class ReportTests(unittest.TestCase):
    """The NotebookLM-style run report assembles from a run's briefs."""

    def test_build_run_report(self):
        import tempfile
        from pathlib import Path
        from dmn import store, report

        tmp = Path(tempfile.mkdtemp())
        c = store.connect(tmp / "db.sqlite")
        store.start_run(c, "runX", command="wander")
        store.add_journal(
            c, "What's a fresh angle on bandits?", "cold", ["arxiv"], {"total": 0.5},
            str(tmp / "a.md"), activity="code_sketch", run_id="runX", fulfillment=1.0,
            grounding=[{"title": "Bandit Algorithms", "url": "http://x", "source": "arxiv"}],
            rabbit_holes=["Thompson sampling priors"],
        )

        class StubLLM:
            name = "stub"

        md_path = report.build_run_report(c, "runX", llm=StubLLM(), journal_dir=tmp)
        self.assertIsNotNone(md_path)
        text = Path(md_path).read_text()
        self.assertIn("## Overview", text)
        self.assertIn("## Findings", text)
        self.assertIn("bandits", text)
        self.assertIn("arxiv", text)  # grounding source surfaced


class FencedExtractionTests(unittest.TestCase):
    """Guards for the 'always the same dry-run app' bug: truncated output must not yield
    None (which made callers fall back to a canned stub)."""

    def test_closed_fence(self) -> None:
        text = "blah\n```python\nprint(1)\n```\ntrailing"
        self.assertEqual(extract_fenced(text, "python"), "print(1)")

    def test_unclosed_fence_returns_none_without_salvage(self) -> None:
        text = "```html\n<!doctype html><html><body>truncated..."
        self.assertIsNone(extract_fenced(text, "html"))

    def test_unclosed_fence_salvaged(self) -> None:
        text = "```html\n<!doctype html><html><body>truncated..."
        got = extract_fenced(text, "html", salvage=True)
        self.assertIsNotNone(got)
        self.assertIn("<body>", got)

    def test_extract_html_from_fence(self) -> None:
        text = "Here is the app:\n```html\n<!doctype html>\n<html></html>\n```"
        self.assertIn("<html>", extract_html(text))

    def test_extract_html_from_raw_truncated_doc(self) -> None:
        text = "Sure!\n<!doctype html>\n<html><head><title>X</title></head><body>cut off"
        got = extract_html(text)
        self.assertIsNotNone(got)
        self.assertTrue(got.lower().startswith("<!doctype html"))

    def test_extract_html_none_when_no_html(self) -> None:
        self.assertIsNone(extract_html("I refuse to write code, sorry."))


if __name__ == "__main__":
    unittest.main()
