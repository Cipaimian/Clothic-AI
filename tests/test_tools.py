"""Tests for the CPU tools: threshold sweep + dataset unifier helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# tools/ is a scripts dir, not a package; add it to the path for import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from threshold_tuner import sweep  # noqa: E402


def test_threshold_sweep_monotonic_recall():
    # As the threshold rises, recall is non-increasing.
    scores = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 1, 1, 1]
    rows = sweep(scores, labels, steps=11)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    # At threshold 0 everything is flagged -> recall 1.0.
    assert rows[0]["recall"] == 1.0


def test_threshold_sweep_with_groups_adds_gap():
    scores = [0.2, 0.8, 0.2, 0.8]
    labels = [0, 1, 0, 1]
    groups = ["A", "A", "B", "B"]
    rows = sweep(scores, labels, groups, steps=5)
    assert "fairness_gap" in rows[0]
