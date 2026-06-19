"""Fit per-head temperature scaling from a labeled validation set.

Input CSV columns: ``head,prob,label`` where ``prob`` is the model's reported
probability (0..1) and ``label`` is the ground truth (0/1). One row per
prediction. Rows are grouped by ``head`` and a temperature is fitted per head.

Output: configs/calibration.json -> picked up automatically by the pipeline.

    python tools/fit_calibration.py --csv data/calib.csv --out configs/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from clothic.reasoning.calibration import expected_calibration_error, fit_temperature


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=Path("configs/calibration.json"), type=Path)
    args = ap.parse_args()

    by_head: dict[str, list[tuple[float, int]]] = defaultdict(list)
    with args.csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_head[row["head"]].append((float(row["prob"]), int(row["label"])))

    temperatures: dict[str, float] = {}
    report: dict[str, dict] = {}
    for head, rows in by_head.items():
        probs = np.array([p for p, _ in rows])
        labels = np.array([l for _, l in rows])
        t = fit_temperature(probs, labels)
        ece_before = expected_calibration_error(probs, labels)
        from clothic.reasoning.calibration import apply_temperature

        ece_after = expected_calibration_error(apply_temperature(probs, t), labels)
        temperatures[head] = round(t, 4)
        report[head] = {"n": len(rows), "T": round(t, 4),
                        "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"temperatures": temperatures}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
