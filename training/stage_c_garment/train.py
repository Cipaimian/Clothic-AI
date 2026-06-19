"""Stage C: fine-tune the campus garment detector (YOLOv11).

Lazy on ultralytics so the repo imports without it. Run::

    pip install "clothic[perception]"
    python training/stage_c_garment/train.py --data training/stage_c_garment/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).with_name("data.yaml")))
    ap.add_argument("--model", default="yolo11s.pt", help="base weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs_clothic/garment")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name="stage_c",
        # Augmentations matched to the documented weaknesses (lighting/occlusion).
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0, mosaic=1.0, erasing=0.4,
        patience=25, cos_lr=True, optimizer="AdamW",
    )
    metrics = model.val(data=args.data)
    print("mAP50-95:", getattr(metrics.box, "map", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
