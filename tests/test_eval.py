"""Tests for the evaluation harness (metrics + fairness)."""

from __future__ import annotations

from clothic.eval.metrics import (
    binary_prf,
    confusion_counts,
    fairness_report,
    false_accusation_rate,
    multilabel_report,
)


def test_confusion_and_prf_perfect():
    yt = [1, 1, 0, 0]
    m = binary_prf(yt, yt)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_prf_known_values():
    # 2 TP, 1 FP, 1 FN
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 1]
    c = confusion_counts(y_true, y_pred)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 0)
    m = binary_prf(y_true, y_pred)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9


def test_false_accusation_rate():
    # compliant(0) wrongly flagged(1): one FP out of two compliant -> 0.5
    y_true = [0, 0, 1]
    y_pred = [1, 0, 1]
    assert false_accusation_rate(y_true, y_pred) == 0.5


def test_multilabel_macro_f1():
    y_true = {"sleeveless": [1, 0, 1], "ripped": [0, 0, 1]}
    y_pred = {"sleeveless": [1, 0, 1], "ripped": [0, 0, 0]}
    rep = multilabel_report(y_true, y_pred)
    assert rep["per_attribute"]["sleeveless"]["f1"] == 1.0
    assert rep["per_attribute"]["ripped"]["f1"] == 0.0
    assert abs(rep["macro_f1"] - 0.5) < 1e-9


def test_fairness_gap_detects_disparity():
    # Group A is judged fairly; group B suffers many false accusations.
    y_true = [0, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1]
    groups = ["A", "A", "A", "B", "B", "B"]
    rep = fairness_report(y_true, y_pred, groups, metric="false_accusation_rate")
    assert rep["per_group"]["A"] == 0.0
    assert rep["per_group"]["B"] == 1.0
    assert rep["gap"] == 1.0
    assert rep["worst_group"] == "B"
