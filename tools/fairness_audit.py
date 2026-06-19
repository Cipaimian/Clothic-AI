"""Audit decision fairness across appearance groups.

Reads predictions + ground truth + a group label per sample and reports each
metric per group plus the worst-case gap -- the number to gate releases on.

Input CSV columns: ``label,pred,group`` (label/pred: 1=violation, 0=compliant).

    python tools/fairness_audit.py --csv data/audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from clothic.eval.metrics import fairness_report

_METRICS = ["false_accusation_rate", "recall", "precision", "f1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--gap-budget", type=float, default=0.1,
                    help="Fail if any metric's group gap exceeds this.")
    args = ap.parse_args()

    labels, preds, groups = [], [], []
    with args.csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
            groups.append(row["group"])

    report = {m: fairness_report(labels, preds, groups, metric=m) for m in _METRICS}
    print(json.dumps(report, indent=2))

    violations = {m: r["gap"] for m, r in report.items() if r["gap"] > args.gap_budget}
    if violations:
        print(f"\nFAIRNESS GATE FAILED (budget {args.gap_budget}): {violations}")
        return 1
    print(f"\nFairness gate passed (all gaps <= {args.gap_budget}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
