# Clothic AI - Model Selection (researched June 2026)

The system is intentionally **two-stage**, matching the requirement:
*parse the body, then check whether each garment covers the scanned parts.*

```
Model 1: BODY PARSING        Model 2: GARMENT
  Meta Sapiens                 detector (YOLO/best.pt) -> garment TYPE
  (Goliath 28-class)           FashionSigLIP          -> garment ATTRIBUTES
        │                              │
        └──────────► COVERAGE / EXPOSURE ◄───────────┘
        exposure(region) = skin_px / (skin_px + clothing_px)
```

## Model 1 - body parsing: **Meta Sapiens** (`perception/parsing.py`)
Why it fits perfectly: Sapiens' 28-class Goliath vocabulary labels **bare anatomy
and clothing as separate classes**. A covered arm is `Upper Clothing` pixels; a
bare arm is `Left/Right Upper Arm` pixels. So coverage is directly measurable:

```
exposure(region) = skin_pixels / (skin_pixels + clothing_pixels)
coverage(region) = 1 - exposure(region)
```

- **Skin classes:** Torso, L/R Upper Arm, L/R Lower Arm, L/R Hand, L/R Upper Leg,
  L/R Lower Leg, L/R Foot, Face Neck.
- **Clothing classes:** Upper Clothing, Lower Clothing, Apparel, Shoe, Sock.
- Region windows (shoulder/upper_arm/midriff/thigh/knee/calf) come from **pose**
  anchors (`perception/pose.py`), so the bands are anatomically placed.
- Sizes: Sapiens 0.3B (lightest) / 1B / 2B; **Sapiens2** 0.4B–5B, native 1K/4K
  (latest, Apr 2026). Checkpoints are TorchScript - pass the path as
  `sapiens_checkpoint`.
- **Fallback:** `SegformerParser` (HF `mattmdjaga/segformer_b2_clothes`) - same
  skin/clothing math, lighter, pip-installable.
- **License:** verify Sapiens' terms before any commercial deployment (fine for
  a thesis / local use).

## Model 2 - garments
- **Type:** your fine-tuned YOLO detector (`legacy/.../best.pt`) maps to canonical
  garment types via `ontology/`.
- **Attributes:** **Marqo-FashionSigLIP** (`perception/attributes.py`) - a SigLIP-2
  model fine-tuned on fashion (+57–78% over FashionCLIP); sigmoid training suits
  multi-label attributes (sleeveless, ripped, sheer, crop, formal). Falls back to
  plain OpenCLIP `ViT-B-32` for low-resource machines.

## What did NOT change
The reasoning core, scoring, rule JSON, and explanations are untouched - both
parsers emit the same `exposure` dict contract. Swapping Model 1/2 is a perception
detail; **policy still decides compliance, not a model.**

## Install
```bash
pip install "clothic[perception]"   # ultralytics + torch (detect/track/pose)
pip install "clothic[parsing]"      # torch + transformers + open_clip_torch (Sapiens + SigLIP)
```

## Sources
- Sapiens - https://learnopencv.com/sapiens-human-vision-models/ · https://huggingface.co/facebook/sapiens
- Sapiens2 - https://www.marktechpost.com/2026/04/27/meta-ai-releases-sapiens2-a-high-resolution-human-centric-vision-model-for-pose-segmentation-normals-pointmap-and-albedo/
- Spectrum (alt.) - https://arxiv.org/abs/2508.06032
- Marqo-FashionSigLIP - https://github.com/marqo-ai/marqo-FashionCLIP
- SigLIP 2 - https://arxiv.org/pdf/2502.14786
