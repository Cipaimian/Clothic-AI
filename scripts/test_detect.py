"""Quick way to SEE the garment detector working on your own images.

Two things it shows:
  1. Raw detections  -> which garments the model found + confidence, and an
     annotated image (boxes drawn) saved so you can eyeball it.
  2. Clothic AI verdict   -> the full compliance decision (compliant / violation,
     which rules fired, what to change) via the real pipeline.  [--verdict]

Usage (PowerShell):
    # detect on the built-in examples (no args)
    python scripts/test_detect.py

    # detect on your own image or a whole folder
    python scripts/test_detect.py "C:/path/to/photo.jpg"
    python scripts/test_detect.py "C:/path/to/folder"

    # also run the full sopan/tidak verdict on one image
    python scripts/test_detect.py "C:/path/to/photo.jpg" --verdict

    # use a specific weights file (otherwise the newest trained best.pt is used)
    python scripts/test_detect.py photo.jpg --model runs/detect/.../weights/best.pt

Annotated images are written to runs/detect/predict*/ - open them to see the boxes.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, "src")  # so `--verdict` can import clothic without PYTHONPATH


def newest_best() -> str | None:
    cands = glob.glob("runs/**/weights/best.pt", recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="examples",
                    help="image file, folder, or '0' for webcam (default: examples/)")
    ap.add_argument("--model", default=None, help="weights .pt (default: newest trained best.pt)")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--verdict", action="store_true", help="also run the full Clothic AI decision")
    args = ap.parse_args()

    model_path = args.model or newest_best()
    if not model_path or not os.path.exists(model_path):
        sys.exit("No trained model found yet. Wait for training to finish, or pass --model.")
    print(f"Model : {model_path}")
    print(f"Source: {args.source}\n")

    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(args.source, conf=args.conf, save=True, verbose=False)

    total = 0
    for r in results:
        name = os.path.basename(getattr(r, "path", "?"))
        if r.boxes is None or len(r.boxes) == 0:
            print(f"  {name:40s} -> (nothing detected)")
            continue
        dets = [f"{r.names[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes]
        total += len(dets)
        print(f"  {name:40s} -> {', '.join(dets)}")
    print(f"\n{total} garments detected across {len(results)} image(s).")
    if results:
        print(f"Annotated images saved in: {results[0].save_dir}  (open them to see the boxes)")

    if args.verdict:
        # One full sopan/tidak verdict through the real pipeline on the first image.
        first = results[0].path if results else args.source
        print("\n--- Clothic AI verdict (full pipeline) ---")
        import cv2

        from clothic.pipeline import ClothicPipeline

        pipe = ClothicPipeline(
            profile_id="default", backend="full",
            backend_kwargs=dict(
                person_weights="yolo11n.pt", garment_weights=model_path,
                pose_weights="yolo11n-pose.pt", parser_type="sapiens",
                sapiens_checkpoint="models/sapiens/sapiens_0.3b_goliath.pt2", device="cpu",
            ),
            enable_temporal=False,
        )
        res = pipe.process_frame(cv2.imread(first))
        for p in res.persons:
            print(f"  decision : {p.decision}")
            up = p.observation.upper.type if p.observation.upper else None
            lo = p.observation.lower.type if p.observation.lower else None
            print(f"  garments : upper={up} lower={lo}")
            if p.matched_rules:
                print(f"  rules    : {', '.join(r.id for r in p.matched_rules)}")
            if p.remediation:
                print(f"  to comply: {'; '.join(p.remediation.steps)}")
        pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
