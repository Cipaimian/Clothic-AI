# Clothic AI deployment

Three topologies (pick per privacy/latency needs):

1. **Edge (per-camera/kiosk)** - NVIDIA Jetson Orin, all models TensorRT-INT8.
   Frames never leave the device (best privacy).
2. **On-prem server** - 1–N GPUs + Triton serving every model, many RTSP
   streams. Recommended for a student-facing system (data stays in-house).
3. **Hybrid** - edge runs detect+track, streams crops to a server for the heavy
   parsing/attribute stages.

## Quantization strategy
| Model | Precision | Note |
|-------|-----------|------|
| Person detector | INT8 (PTQ + calibration set) | big speedup, tiny mAP loss |
| Pose | FP16 (INT8 if validated) | keypoint precision sensitive |
| Parsing (SegFormer) | FP16 | INT8 can hurt thin-region IoU; QAT if needed |
| Garment det/seg | INT8 + QAT fallback | re-check mask IoU after PTQ |
| CLIP attribute backbone | FP16 | keep embeddings stable |

**Always re-check calibration (ECE) after quantizing** - quantization can
de-calibrate confidences. Re-run `tools/fit_calibration.py` on the quantized
models and commit the new `configs/calibration.json`.

## Export to TensorRT
```bash
pip install "clothic[perception]"
python deploy/tensorrt/export.py --weights legacy/runs/detect/train/weights/best.pt \
    --format engine --half          # FP16; add --int8 --data <calib.yaml> for INT8
```

## Triton
`deploy/triton/model_repository/` holds the layout. Point Triton at it:
```bash
tritonserver --model-repository=deploy/triton/model_repository
```
Enable dynamic batching + concurrent model execution per `config.pbtxt`.

## Edge (Jetson) checklist
- Build TensorRT engines on-device (engines are device-specific); cache them.
- Cap the heavy-stage cadence (`configs/pipeline.yaml: cadence.semantic_every`)
  to fit the power/thermal budget; use dynamic resolution.
- Offline-first: local policy cache + store-and-forward events when the network
  is down (the SQLite `EventStore` already supports this).
- Sign model artifacts; keep field policy read-only; keep the audit log.

## Observability
Prometheus + Grafana (latency p50/p95, FPS, GPU mem, queue depth, per-stage
drop rate). Gate any model promotion on **false-accusation rate + fairness gap**
vs the incumbent (shadow-mode canary).
