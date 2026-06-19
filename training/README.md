# Clothic AI training

Each perception model is trained **independently** (modular), then the assembled
system is calibrated and evaluated end-to-end. Requires the heavy deps:

```bash
pip install "clothic[perception]"           # ultralytics + torch
pip install transformers open_clip_torch albumentations  # parsing + attributes
```

## Stages
| Stage | Dir | Model | Data |
|-------|-----|-------|------|
| A | (pretrained) | person detect + pose | COCO / CrowdHuman |
| B | `stage_b_parsing/` | SegFormer-B2 | LIP → ATR → CIHP → campus |
| C | `stage_c_garment/` | YOLOv11-seg / RT-DETR | DeepFashion2 → campus (Roboflow) |
| D | `stage_d_attributes/` | CLIP probes | Fashionpedia → campus |
| E | (offline) | temperature scaling | held-out val (`tools/fit_calibration.py`) |
| F | (offline) | system eval | in-situ campus test (`clothic.eval`) |

## Recommended recipe (all stages)
- Transfer learning: start from COCO/ImageNet/CLIP weights; freeze backbone,
  unfreeze gradually (discriminative LR).
- Losses: detection = CIoU+CE; parsing = CE + Dice/Lovász; attributes
  (imbalanced multi-label) = **focal / asymmetric loss**.
- AdamW + cosine schedule + warmup + EMA + mixed precision.
- Augment with Albumentations (bbox/mask/keypoint-aware): brightness/CLAHE,
  CoarseDropout (occlusion), affine/perspective, motion blur + JPEG (camera
  realism), Mosaic for detection. Avoid vertical flip / heavy color shift that
  destroys skin/transparency cues.
- Track every run (config, git SHA, dataset DVC hash, metrics, fairness) in
  MLflow/W&B.

## Quick start (stage C - fine-tune the campus garment detector)
```bash
python training/stage_c_garment/train.py \
    --data training/stage_c_garment/data.yaml --epochs 100 --imgsz 640
```
