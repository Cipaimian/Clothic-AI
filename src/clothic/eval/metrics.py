"""Metrics for Clothic AI evaluation.

Covers the three things that matter for an explainable, fair compliance system:
  * per-attribute classification quality (precision / recall / F1, macro),
  * the headline **false-accusation rate** (compliant judged as violation),
  * **fairness gaps** -- the spread of a metric across appearance groups, which
    is the number a release should be gated on, not just the mean.

Pure numpy; no ML framework needed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """TP/FP/FN/TN for binary labels (1 = positive)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(np.sum((yt == 1) & (yp == 1))),
        "fp": int(np.sum((yt == 0) & (yp == 1))),
        "fn": int(np.sum((yt == 1) & (yp == 0))),
        "tn": int(np.sum((yt == 0) & (yp == 0))),
    }


def binary_prf(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Precision, recall, F1, accuracy for a binary task."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, **c}


def multilabel_report(
    y_true: dict[str, Sequence[int]], y_pred: dict[str, Sequence[int]]
) -> dict[str, object]:
    """Per-attribute PRF plus macro-F1 (handles class imbalance).

    ``y_true``/``y_pred`` map attribute name -> 0/1 sequence over samples.
    """
    per_attr = {name: binary_prf(y_true[name], y_pred[name]) for name in y_true}
    macro_f1 = float(np.mean([m["f1"] for m in per_attr.values()])) if per_attr else 0.0
    return {"per_attribute": per_attr, "macro_f1": macro_f1}


def false_accusation_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of truly-compliant people flagged as violation (FP rate).

    The most harmful error in a dress-code system; report and minimise it.
    Here 1 = violation, 0 = compliant.
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["fp"] + c["tn"]
    return c["fp"] / denom if denom else 0.0


def fairness_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    metric: str = "false_accusation_rate",
) -> dict[str, object]:
    """Compute a metric per appearance group and the worst-case gap.

    ``groups`` labels each sample (e.g. skin-tone bucket). The **gap**
    (max - min across groups) is the fairness number to gate releases on.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups)

    metric_fns = {
        "false_accusation_rate": false_accusation_rate,
        "recall": lambda a, b: binary_prf(a, b)["recall"],
        "precision": lambda a, b: binary_prf(a, b)["precision"],
        "f1": lambda a, b: binary_prf(a, b)["f1"],
        "accuracy": lambda a, b: binary_prf(a, b)["accuracy"],
    }
    if metric not in metric_fns:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(metric_fns)}")
    fn = metric_fns[metric]

    per_group: dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        per_group[str(g)] = float(fn(yt[mask], yp[mask]))

    values = list(per_group.values())
    gap = (max(values) - min(values)) if values else 0.0
    worst = max(per_group, key=per_group.get) if per_group else None
    return {
        "metric": metric,
        "per_group": per_group,
        "gap": gap,
        "worst_group": worst,
        "overall": float(fn(yt, yp)),
    }
