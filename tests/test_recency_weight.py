"""Tests for recency-weighted composite taste weights."""
from __future__ import annotations

import math

import pytest

from dmn import taste


def test_manual_source_max_recency():
    w = taste.composite_weight(visit_count=3, days_since=400, source="manual")
    assert w == math.log1p(3) * 1.0


def test_interview_source_max_recency():
    w = taste.composite_weight(visit_count=1, days_since=999, source="interview")
    assert w == math.log1p(1) * 1.0


def test_journal_import_slight_discount():
    w = taste.composite_weight(visit_count=2, days_since=0, source="journal_import")
    browser = taste.composite_weight(visit_count=2, days_since=0, source="browser:arc")
    assert w == pytest.approx(math.log1p(2) * 0.95)
    assert w < browser


def test_old_browser_row_lower_than_manual():
    old_browser = taste.composite_weight(5, 200.0, "browser:chrome")
    manual = taste.composite_weight(5, 200.0, "manual")
    assert manual > old_browser
