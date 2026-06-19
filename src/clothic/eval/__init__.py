"""Evaluation harness: detection/attribute metrics + fairness auditing."""

from clothic.eval.metrics import (
    binary_prf,
    confusion_counts,
    fairness_report,
    multilabel_report,
)

__all__ = ["binary_prf", "confusion_counts", "fairness_report", "multilabel_report"]
