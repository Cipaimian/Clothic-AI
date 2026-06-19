"""Export a YOLO model to TensorRT/ONNX for deployment.

Lazy on ultralytics. INT8 needs a calibration dataset (a data.yaml whose images
are representative campus footage).

    python deploy/tensorrt/export.py --weights best.pt --format engine --half
    python deploy/tensorrt/export.py --weights best.pt --format engine --int8 --data calib.yaml
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="engine", choices=["engine", "onnx", "openvino"])
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--int8", action="store_true", help="INT8 (needs --data)")
    ap.add_argument("--data", default=None, help="calibration data.yaml for INT8")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if args.int8 and not args.data:
        ap.error("--int8 requires --data (a calibration dataset)")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    path = model.export(
        format=args.format, half=args.half, int8=args.int8,
        data=args.data, imgsz=args.imgsz,
    )
    print(f"Exported -> {path}")
    print("Reminder: re-run tools/fit_calibration.py - quantization can shift ECE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
