"""Real-time webcam demo - live garment + compliance overlay from your camera.

Uses the FAST `ultralytics` backend (person detect + ByteTrack + garment detect +
attribute-derived exposure) so it runs in real time on CPU. The heavy Sapiens
pixel-exposure path is single-image only (~30 s/frame) and is NOT used here.

    python scripts/demo_webcam.py                 # camera 0, newest trained model
    python scripts/demo_webcam.py --source 1       # another camera
    python scripts/demo_webcam.py --conf 0.3       # more sensitive

Press 'q' to quit. Boxes are coloured by decision; the label shows the detected
garments + verdict.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")

from clothic.pipeline import ClothicPipeline
from clothic.schemas import Decision

_COLORS = {
    Decision.COMPLIANT: (0, 170, 0),
    Decision.MINOR_VIOLATION: (0, 165, 255),
    Decision.MAJOR_VIOLATION: (0, 0, 220),
    Decision.INSUFFICIENT_EVIDENCE: (150, 150, 150),
}


def newest_best() -> str | None:
    cands = glob.glob("runs/**/weights/best.pt", recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None


def draw(frame, result) -> None:
    for p in result.persons:
        x, y, w, h = (int(v) for v in p.bbox)
        color = _COLORS[p.decision]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        ov = p.scores.overall_violation
        tag = f"#{p.track_id} {p.decision.value}" + (f" {ov:.2f}" if ov is not None else "")
        # Only list garments when we actually made a decision. While abstaining
        # (insufficient_evidence) the evidence is unreliable, so don't claim a
        # garment label that might be wrong.
        garments = [g.type for g in (p.observation.upper, p.observation.lower,
                                     p.observation.footwear) if g]
        show_sub = p.decision is not Decision.INSUFFICIENT_EVIDENCE and garments
        h_lbl = 40 if show_sub else 22
        cv2.rectangle(frame, (x, y - h_lbl), (x + max(170, 8 * len(tag)), y), color, -1)
        cv2.putText(frame, tag, (x + 4, y - (24 if show_sub else 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if show_sub:
            cv2.putText(frame, ", ".join(garments), (x + 4, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    fps = 1000.0 / max(result.latency_ms.get("total", 1), 1)
    cv2.putText(frame, f"Clothic AI  {result.latency_ms.get('total', 0):.0f}ms  ~{fps:.1f} FPS",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--person", default="yolo11n.pt")
    ap.add_argument("--garment", default=None, help="garment weights (default: newest trained best.pt)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--zone", default=None)
    args = ap.parse_args()

    garment = args.garment or newest_best()
    if not garment:
        sys.exit("No trained garment model found. Train first or pass --garment.")
    print(f"person : {args.person}\ngarment: {garment}\n(press 'q' in the window to quit)")

    # Real-time path: temporal smoothing ON (debounce stabilises the live verdict).
    pipe = ClothicPipeline(
        profile_id=args.profile, backend="ultralytics", zone=args.zone,
        backend_kwargs=dict(person_weights=args.person, garment_weights=garment, conf=args.conf),
    )
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/source {source!r}.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipe.process_frame(frame)
        draw(frame, result)
        cv2.imshow("Clothic AI - realtime (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
