"""Sweep a decision threshold and report the precision/recall trade-off.

Lets an administrator pick an operating point *consciously* -- in a dress-code
system you typically favour a low **false-accusation rate** (compliant judged
as violation) even at some cost to recall.

Input CSV columns: ``violation_score,label`` (label 1 = truly violating).
Optional ``group`` column enables a fairness column in the sweep.

    python tools/threshold_tuner.py --csv data/scores.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from clothic.eval.metrics import binary_prf, fairness_report, false_accusation_rate


def sweep(scores, labels, groups=None, steps=21):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []
    for thr in np.linspace(0.0, 1.0, steps):
        pred = (scores >= thr).astype(int)
        m = binary_prf(labels, pred)
        row = {
            "threshold": round(float(thr), 3),
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
            "false_accusation_rate": round(false_accusation_rate(labels, pred), 3),
        }
        if groups is not None:
            fr = fairness_report(labels, pred, groups, metric="false_accusation_rate")
            row["fairness_gap"] = round(fr["gap"], 3)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=21)
    args = ap.parse_args()

    scores, labels, groups = [], [], []
    with args.csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        has_group = "group" in (reader.fieldnames or [])
        for row in reader:
            scores.append(float(row["violation_score"]))
            labels.append(int(row["label"]))
            if has_group:
                groups.append(row["group"])

    rows = sweep(scores, labels, groups or None, steps=args.steps)
    headers = list(rows[0].keys())
    print(" | ".join(h.rjust(12) for h in headers))
    for r in rows:
        print(" | ".join(str(r[h]).rjust(12) for h in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
