# Clothic AI - Architecture Specification

A real-time campus dress-code compliance system built as a
**Visual Attribute Recognition + Policy Reasoning Engine**, not a binary
"sopan / tidak sopan" classifier.

> Core design principle: **the neural networks never learn "modesty".**
> They only detect *observable, defensible visual attributes* (garment type,
> sleeve length, hemline relative to knee, transparency cues, rips, skin
> exposure ratio per body region). A **transparent, auditable rule engine**
> - driven by editable JSON policy profiles - maps those attributes to a
> compliance decision with a written explanation. This makes the system
> explainable, contestable, configurable per campus, and far less biased
> than training a model on a subjective label.

---

## Table of Contents
1. [Architecture Redesign](#1-architecture-redesign)
2. [AI Pipeline (stage-by-stage)](#2-ai-pipeline-stage-by-stage)
3. [Recommended Models](#3-recommended-models)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Training Pipeline](#5-training-pipeline)
6. [Real-Time Inference Pipeline](#6-real-time-inference-pipeline)
7. [Explainability System](#7-explainability-system)
8. [Rule Engine Design](#8-rule-engine-design)
9. [JSON Rule Examples](#9-json-rule-examples)
10. [Example API Responses](#10-example-api-responses)
11. [Database Schema](#11-database-schema)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Deployment Strategy](#13-deployment-strategy)
14. [Fairness, Reliability & Confidence Calibration](#14-fairness-reliability--confidence-calibration)
15. [Folder Structure](#15-folder-structure)
16. [Future Improvements Roadmap](#16-future-improvements-roadmap)

---

## 1. Architecture Redesign

Clothic AI is a **modular, staged perception → reasoning → presentation pipeline**.
Each stage is an independent, swappable component with a typed contract, so any
model can be upgraded without touching the rest of the system.

```
                          ┌──────────────────────────────────────────────┐
                          │            CONFIG / POLICY LAYER              │
                          │  campus_profiles/*.json  ·  thresholds.json   │
                          │  model_registry.yaml     ·  weights.json      │
                          └───────────────┬──────────────────────────────┘
                                          │ (hot-reloadable)
   FRAME                                  ▼
   SOURCE     ┌─────────────────────────────────────────────────────────────┐
  (webcam/    │                    PERCEPTION CORE                           │
   RTSP/file) │                                                             │
     │        │  S1 Ingest/Preprocess ─► S2 Person Detect ─► S3 Track       │
     └───────►│         │                       │               │           │
              │         ▼                       ▼               ▼           │
              │  S4 Pose Estimation     S5 Human Parsing   (track_id, bbox) │
              │         │                       │                           │
              │         └───────────┬───────────┘                           │
              │                     ▼                                       │
              │            S6 Garment Detection / Classification            │
              │                     │                                       │
              │                     ▼                                       │
              │   S7 Attribute Classification (multi-label per garment)     │
              │                     │                                       │
              │                     ▼                                       │
              │   S8 Exposure Estimation (skin ratio per body region)       │
              │                     │                                       │
              │                     ▼                                       │
              │   S9 Confidence Fusion + Temporal Smoothing (per track)     │
              └─────────────────────┬───────────────────────────────────────┘
                                    ▼
              ┌─────────────────────────────────────────────────────────────┐
              │                  REASONING CORE                              │
              │   S10 Rule Engine (JSON policy)  ─►  S11 Scoring             │
              │        · evaluates predicates over the attribute vector      │
              │        · weighted, calibrated, profile-driven                │
              └─────────────────────┬───────────────────────────────────────┘
                                    ▼
              ┌─────────────────────────────────────────────────────────────┐
              │               EXPLAINABILITY + PRESENTATION                  │
              │  S12 Explanation generator (templated NL + evidence map)     │
              │  S13 Outputs: REST/WebSocket API · Dashboard UI · Audio ·    │
              │       annotated video · audit log · DB persistence           │
              └─────────────────────────────────────────────────────────────┘
```

**Why this shape**
- **Separation of perception and policy.** Models output *facts*; policy decides
  *meaning*. You can change campus rules at runtime with zero retraining.
- **Every decision is traceable** to (a) which attributes fired, (b) their
  confidences, (c) which rule matched, (d) the weight applied.
- **Graceful degradation.** If parsing fails (occlusion), the fusion stage
  lowers confidence and the rule engine returns `INSUFFICIENT_EVIDENCE`
  instead of a false violation.

---

## 2. AI Pipeline (stage-by-stage)

| Stage | Name | Input | Output | Default model |
|-------|------|-------|--------|---------------|
| S1 | Ingest / Preprocess | raw frame | normalized tensor, letterboxed | OpenCV / DALI |
| S2 | Person detection | frame | person boxes + conf | RT-DETR-R50 or YOLOv11-m |
| S3 | Tracking | boxes over time | stable `track_id` | ByteTrack |
| S4 | Pose estimation | person crop | 17 COCO keypoints + conf | RTMPose-m (or ViTPose) |
| S5 | Human parsing | person crop | per-pixel body/garment mask (20 classes) | SegFormer-B2 fine-tuned on LIP/ATR/CIHP |
| S6 | Garment detection | person crop | garment instances (bbox/mask + type) | YOLOv11-seg or RT-DETR on DeepFashion2 |
| S7 | Attribute classification | garment crop | multi-label attribute vector | CLIP (ViT-L/14) + linear probe, or Swin multi-head |
| S8 | Exposure estimation | parsing mask + pose + skin mask | skin-exposure ratio per region | derived (skin seg + geometry) |
| S9 | Fusion + temporal | per-frame evidence | smoothed per-track attribute vector + uncertainty | EMA / Bayesian filter |
| S10 | Rule engine | attribute vector + policy | matched rules + raw scores | rules-py engine |
| S11 | Scoring | matched rules | exposure/formality/compliance/uncertainty scores | weighted aggregator |
| S12 | Explainability | rules + evidence | NL explanation + evidence overlay | template + (optional) LLM polish |
| S13 | Output | decision object | API/UI/audio/log | FastAPI + WebSocket |

### Why three perception signals feed exposure (S8)
Exposure is **never** a single model's guess. It is computed geometrically and
is therefore auditable:

- **Human parsing (S5)** gives which pixels are skin vs garment vs background,
  segmented into body regions (torso, upper-arm, forearm, thigh, calf).
- **Pose (S4)** gives anatomical landmarks. The **knee keypoint** defines the
  "above/below knee" line; the **shoulder/elbow** keypoints define sleeve
  coverage. Hemline = (lowest garment pixel on the leg) vs (knee y-coordinate),
  normalized by limb length so it is scale- and distance-invariant.
- **Skin segmentation** (a lightweight class inside the parsing head)
  confirms exposed-skin pixels rather than skin-colored fabric.

`exposure_ratio(region) = skin_pixels(region) / total_pixels(region)` - a number
you can show, threshold, and defend in an appeal. No black box decides "too much
skin"; the policy threshold does, and it is visible in JSON.

### Attribute taxonomy (what S6/S7 actually produce)
The classes you listed, reorganized into **type** (mutually exclusive, S6) and
**attributes** (multi-label, S7) so a single garment can carry several flags.

```
GARMENT TYPE (upper):  tshirt | polo | tank_top | sleeveless_top | crop_top |
                       hoodie | jacket | shirt_formal | blouse | dress | other
GARMENT TYPE (lower):  long_pants | shorts | skirt_long | skirt_short |
                       trousers_formal | leggings | other
FOOTWEAR:              shoes_closed | sneakers | sandals | flip_flops |
                       formal_shoes | none_barefoot
ATTRIBUTES (per garment, multi-label, 0..1 each):
   sleeveless · short_sleeve · long_sleeve
   midriff_exposed (crop) · transparent_sheer · ripped_torn
   hemline_above_knee · hemline_at_knee · hemline_below_knee
   tight_fit · graphic_print · formal_style
EXPOSURE (per region, derived 0..1):
   shoulder · upper_arm · midriff · thigh · knee · calf · cleavage_proxy
```

> **Important honesty note on two of your requested classes.**
> *"Transparent clothing indication"* and *"overly revealing clothing
> indicators"* are kept as **soft, low-weight, advisory signals**, never as
> hard auto-violations. Sheerness is genuinely hard to detect reliably from a
> single webcam frame (depends on lighting/backlight), and "overly revealing"
> is exactly the kind of subjective judgment we deliberately push into policy.
> Clothic AI detects *measurable proxies* (e.g., visible-skin-through-fabric edge
> response, neckline depth ratio) and surfaces them as **"flag for human
> review"** rather than asserting a violation. This is the responsible and
> academically defensible choice.

---

## 3. Recommended Models

Picked for the **accuracy / real-time / license** trade-off, with fallbacks.

| Role | Primary | Why | Alternatives |
|------|---------|-----|--------------|
| Person detection | **RT-DETR-R50** (Apache-2.0) | NMS-free, very stable boxes, great for crowded campus scenes; strong real-time on GPU | YOLOv11-m (fast, easy), YOLOv11-x (max accuracy) |
| Tracking | **ByteTrack** | Associates low-confidence boxes too → keeps IDs through occlusion; no extra ReID net needed for fixed cameras | DeepSORT/BoT-SORT (add ReID if cameras overlap) |
| Pose | **RTMPose-m** (MMPose) | Top speed/accuracy ratio, robust to partial occlusion; gives the knee/shoulder anchors exposure needs | ViTPose (higher acc), OpenPose (legacy, slower) |
| Human parsing | **SegFormer-B2** fine-tuned on LIP/ATR/CIHP | Transformer segmentation, robust to lighting/pose; clean per-region masks | Self-Correction-Human-Parsing (SCHP, strong on LIP), Mask2Former, Detectron2 (PointRend) |
| Garment detection | **YOLOv11-seg** or **RT-DETR** fine-tuned on **DeepFashion2** | Instance masks per garment → feeds attribute crops & exposure | Detectron2 Mask R-CNN, Fashionpedia models |
| Attributes | **CLIP ViT-L/14 + linear probe** per attribute | Zero/few-shot friendly, semantically rich, excellent for rare attributes (ripped/sheer) with little data; text-promptable | Swin-B multi-head classifier, OpenCLIP, SigLIP |
| Open-vocab fallback | **GroundingDINO** | Text-prompted detection of novel items ("crop top", "ripped jeans") when you have no labels yet - great for bootstrapping data & new campus rules | Grounded-SAM (adds masks), OWLv2 |
| Skin region | small head inside SegFormer (skin class) | reuse parsing compute; avoids a separate skin net | dedicated U-Net skin segmenter |

**Design rationale for CLIP-based attributes:** your scarce, subjective
attributes (ripped, sheer, crop, "formal") are exactly where supervised heads
starve for data. A frozen CLIP backbone + a tiny trainable linear probe per
attribute gets you usable accuracy from a few hundred examples, is cheap to
extend (add a new attribute = train one linear layer), and the text-prompt
nature is itself a form of explainability.

---

## 4. Dataset Strategy

### 4.1 Public datasets to assemble

| Dataset | What it gives you | License notes | Used for |
|---------|-------------------|---------------|----------|
| **DeepFashion2** | 13 garment categories, bbox + mask + landmarks + style, 491K images | Research-only; check terms | Garment detection (S6) backbone |
| **Fashionpedia** | 46 apparel categories + **fine-grained attributes** (sleeve length, length, opacity, fit) - *exactly your attribute taxonomy* | CC; verify | Attribute classifier (S7) - the single best fit |
| **iMaterialist Fashion** | Attribute-rich, fine-grained apparel tags | Kaggle terms | Attribute augmentation |
| **LIP** (Look-Into-Person) | 20-class human parsing incl. arms/legs/upper/lower clothes | Research | Human parsing (S5) |
| **ATR** | Human parsing, 18 classes, full-body | Research | Parsing pretrain |
| **CIHP** | Multi-person instance parsing | Research | Parsing in crowds |
| **ModaNet** | Street fashion polygons (13 classes) | Research | Detection/seg aux |
| **Clothing Co-Parsing (CCP)** | Pixel clothing labels | Research | Parsing aux |
| **COCO** (person) | Person boxes + 17 keypoints | CC BY 4.0 | Person detect + pose pretrain |
| **CrowdHuman** | Dense person boxes w/ occlusion | Research | Robust person detect |
| Your **Roboflow set** (388 imgs, 4 cls) | Indonesian campus context (kaos/singlet/celana) | CC BY 4.0 | Campus fine-tune seed |

> Always re-verify each license before training a model you intend to deploy.
> Several fashion sets are **research-only** - fine for a thesis/prototype, but
> flag any commercial rollout for legal review.

### 4.2 How to merge heterogeneous datasets

The datasets disagree on label names and granularity. Solve with a **unified
ontology + mapping layer**, never by retraining on raw mixed labels.

```
ontology/
  taxonomy.yaml          # canonical types + attributes (single source of truth)
  mappings/
    deepfashion2.yaml    # df2 "short sleeve top" -> {type: tshirt, attr: short_sleeve}
    fashionpedia.yaml    # fp "sleeve: sleeveless" -> attr: sleeveless
    lip.yaml             # lip "upper-clothes" -> region: upper
    roboflow_campus.yaml # kaos -> tshirt, singlet -> tank_top/sleeveless ...
```

- Write a `dataset_unifier` that ingests each source, applies its mapping, and
  emits a **single normalized format** (COCO-style JSON + a parallel attribute
  table). Garments unmapped in a source become `ignore` (masked from loss), not
  a wrong label.
- Keep **provenance** on every annotation (`source`, `orig_label`) so you can
  audit and re-map later.

### 4.3 Cleaning noisy labels
- **Cross-model agreement filter:** run GroundingDINO + your current model;
  keep boxes where IoU and class agree, queue disagreements for human review.
- **Confident learning / loss-based mining:** flag samples with persistently
  high loss after a few epochs as likely mislabeled; review the top-k.
- **Embedding outlier detection:** CLIP-embed each garment crop; within a class,
  flag far-from-centroid items as candidate label errors or out-of-distribution.
- **Dedup:** perceptual hash (pHash) to drop near-duplicates that would inflate
  metrics - important since Roboflow exports often contain augmented dupes.

### 4.4 Avoiding bias (this is a fairness-critical system)
- **Balance across appearance:** stratify and audit by **skin tone (Fitzpatrick
  proxy), body type, gender presentation, garment color, lighting**. Report
  per-group counts; oversample / targeted-collect the underrepresented.
- **Never train on the protected attribute.** No gender/ethnicity classifier in
  the pipeline. Decisions depend only on garment geometry, which is the legally
  and ethically defensible basis.
- **Decouple skin tone from exposure.** Exposure uses *parsing + pose geometry*,
  not raw skin-color thresholds (which fail on dark skin and false-fire on
  light skin). Validate exposure accuracy **per skin-tone bucket** (see §14).
- **Context diversity:** indoor/outdoor, day/night, backpack/occlusion,
  sitting/standing, hijab/headwear (must not be flagged), winter layers.

### 4.5 Campus-specific fine-tuning data (the part that actually matters)
1. **Define the policy first**, then collect to it. The campus rulebook tells
   you which attributes need high accuracy (e.g., if sandals are allowed,
   footwear precision matters less).
2. **Consented capture sessions:** recruit student volunteers, IRB/ethics
   approval, signed consent, compensate. Capture the real cameras, angles,
   lighting, and the actual garments students wear locally.
3. **Active learning loop:** deploy in shadow mode → log low-confidence /
   high-disagreement frames → human-label those → retrain. This is 10× more
   efficient than random labeling.
4. **Synthetic augmentation for rare cases:** use diffusion inpainting or 3D
   (e.g., dressed SMPL avatars) to generate hard cases (sheer, ripped, extreme
   crop) you can't ethically stage - clearly tag as synthetic, keep <20% of any
   class, and validate that real-set metrics don't regress.
5. **Hold out a real, in-situ test set** from a *different week/camera* than
   training to get an honest generalization number.

---

## 5. Training Pipeline

Train each perception model **independently** (modular), then calibrate and
evaluate the assembled system end-to-end.

```
stage_a_person_pose/   # mostly use pretrained; light fine-tune on CrowdHuman+campus
stage_b_parsing/       # SegFormer-B2: pretrain LIP+ATR+CIHP -> fine-tune campus
stage_c_garment/       # YOLOv11-seg/RT-DETR: DeepFashion2 -> fine-tune campus ontology
stage_d_attributes/    # CLIP frozen + per-attribute linear probes on Fashionpedia+campus
stage_e_calibration/   # temperature scaling per head on a held-out set
stage_f_system_eval/   # end-to-end on in-situ campus test set
```

### Recipe highlights
- **Transfer learning everywhere.** Start from COCO/ImageNet/CLIP weights.
  Freeze backbones early, unfreeze gradually (discriminative LR: low LR for
  backbone, higher for heads).
- **Loss functions:**
  - Detection: standard (CIoU + classification) for YOLO/RT-DETR.
  - Parsing: cross-entropy + Lovász-Softmax or Dice for thin regions (arms).
  - Attributes (multi-label, imbalanced): **focal loss** or
    **asymmetric loss (ASL)** - handles "ripped" appearing in 2% of images.
  - Class imbalance: per-class weights + oversampling rare attributes.
- **Optimizer:** AdamW, cosine schedule, warmup, EMA weights, mixed precision
  (AMP/bf16), gradient clipping.
- **Multi-task option (later):** a shared backbone with parsing + attribute
  heads (à la a fashion multi-task net) to cut latency - only after single-task
  baselines are solid.

### Augmentation (matched to your weaknesses)
| Weakness it fixes | Augmentations |
|---|---|
| Lighting | brightness/contrast/gamma, CLAHE, color jitter, simulated backlight, white-balance shift |
| Occlusion | random erasing, CoarseDropout, cutout over limbs, synthetic backpack/person overlays |
| Pose/scale | affine, perspective, scale jitter, **Mosaic** (detection), copy-paste |
| Camera realism | motion blur, JPEG compression, ISO noise, downscale-upscale, lens distortion |
| Robust attributes | hue/sat shifts so color ≠ class; RandAugment |
| **Not** to use | horizontal flip is fine; avoid vertical flip & heavy color shift that destroys "transparent/skin" cues |

Use **Albumentations** with bbox/mask/keypoint-aware transforms so geometry
labels stay consistent.

### Reproducibility
- Track with **MLflow** or **Weights & Biases**: every run logs config, git SHA,
  dataset version (DVC hash), metrics, calibration curves, per-group fairness.
- **DVC** for dataset + model versioning; deterministic seeds; pinned env.

---

## 6. Real-Time Inference Pipeline

### Execution model
A **staged async pipeline** with per-stage worker pools so the slow stages
(parsing, attributes) don't stall capture.

```
[Capture thread] → frame queue → [Detector+Tracker (batched, GPU)]
   → per-track crop queue → [Parsing+Pose (batched)] → [Garment+Attr (batched)]
   → [Fusion/Temporal] → [Rule engine (CPU)] → [Output/UI/audio]
```

- **Batch across tracks**, not just frames: collect all person crops in a frame
  and run parsing/attributes as one GPU batch.
- **Run cadence (cost control):** detection+tracking every frame; the heavy
  semantic stack (parsing/garment/attributes) every **N-th frame or on
  significant motion**, because clothing doesn't change frame-to-frame. Temporal
  fusion fills the gaps. This is the single biggest real-time win.
- **Region-of-interest:** only process new/changed tracks; cache last attribute
  vector per `track_id`.

### Temporal consistency (S9)
Per `track_id`, maintain a running estimate per attribute:

```
a_t = α · a_obs + (1−α) · a_{t−1}        # EMA smoothing
confidence rises when consecutive frames agree (Bayesian update)
decision only emitted after K-of-M frames exceed threshold (debounce)
```

This kills the flicker that plagues frame-by-frame YOLO systems and prevents a
single bad frame from triggering a false audio warning. Track-level decisions
also mean **one alert per person**, not one per frame.

### Confidence fusion (S9) - multi-stage scoring
Each downstream score is conditioned on the reliability of the evidence:

```
attribute_conf  = model_softmax · calibration_factor
evidence_quality = f(detection_conf, parsing_iou_self, pose_visibility,
                     crop_resolution, occlusion_ratio, motion_blur)
fused_conf       = attribute_conf · evidence_quality · temporal_agreement
```

If `evidence_quality` is low (heavy occlusion, tiny/blurry crop), `fused_conf`
drops and the rule engine routes to `INSUFFICIENT_EVIDENCE` → **no auto-alarm,
optional human review** - directly fixing your occlusion/lighting/pose
robustness weakness.

### GPU optimization
- **TensorRT** engines (FP16, INT8 where safe) for every model; **ONNX Runtime**
  as the portable fallback.
- **NVIDIA Triton Inference Server** to host all models, with dynamic batching
  and concurrent model execution on one GPU.
- **CUDA graphs** for fixed-shape stages; **NVDEC/DALI** for hardware video
  decode + GPU preprocessing (no CPU↔GPU thrash).
- Pin memory, reuse buffers, async CUDA streams; keep crops on-GPU between
  stages.

---

## 7. Explainability System

Every decision yields a structured **evidence object** and a **human-readable
explanation**, both derived from the same source so they can never disagree.

### Layers of explanation
1. **Attribute evidence map** - each fired attribute with its fused confidence
   and the pixels/region that produced it (parsing mask + pose anchor).
2. **Rule trace** - which policy rules matched, their weights, and arithmetic.
3. **Natural-language summary** - template-filled, deterministic, auditable:

```
"Detected SLEEVELESS upper garment (tank_top, conf 0.91) with exposed
 shoulders (exposure 0.78 > policy 0.40) and SHORTS with hemline above the
 knee (knee-ratio 0.32, policy requires ≥ 0.95 coverage).
 Two rules contributed. Compliance violation score: 0.87 (HIGH).
 Evidence quality: 0.84 (good). Action: flag for human review."
```

4. **Visual overlay** - annotated frame: person box, garment masks, the
   knee/shoulder lines that defined the measurement, and color-coded regions.

> The NL layer is **template-based by default** (deterministic, no
> hallucination, runs offline). An LLM may *optionally* rephrase for tone in
> reports, but never invents facts - it only re-words the structured evidence.

### Why templated, not learned, explanations
A learned "explanation model" can hallucinate justifications. Because Clothic AI's
decision is already symbolic (rule engine over measured attributes), the honest
explanation is literally the rule trace rendered to text. That is the academic
gold standard for contestable automated decisions.

---

## 8. Rule Engine Design

A small, dependency-light **predicate evaluation engine** over the per-track
attribute vector. Policies are **data, not code**.

### Concepts
- **Profile** - a named campus policy (e.g., `default`, `lab_safety`,
  `exam_formal`, `sports_facility`). Selectable per camera/zone/time.
- **Rule** - `{ id, description, when (predicate), weight, severity, category,
  citation }`. `citation` points to the handbook clause for appeals.
- **Predicate** - boolean expression over attributes/exposures/scores with
  operators (`>`, `<`, `==`, `and`, `or`, `not`, `exists`, thresholds).
- **Scoring** - weighted aggregation → four sub-scores + calibrated overall.

### Four-score methodology (replaces binary sopan/tidak)
```
exposure_score    = weighted aggregate of region exposures vs policy limits   [0..1]
formality_score   = function of garment formality attrs + footwear            [0..1]
compliance_score  = 1 − normalized(Σ matched_rule.weight · severity)          [0..1]
uncertainty_score = 1 − mean(fused_conf of contributing evidence)             [0..1]
overall_decision  = band(compliance_score, uncertainty_score)
```

Decision bands (configurable):
```
compliant            : compliance ≥ 0.80  and uncertainty ≤ 0.30
minor_violation      : 0.50 ≤ compliance < 0.80
major_violation      : compliance < 0.50  and uncertainty ≤ 0.30
insufficient_evidence: uncertainty > 0.30  (→ no auto-alarm; human review)
```

### Threshold tuning system
- All thresholds live in `thresholds.json`, hot-reloadable.
- A `tuning/` tool sweeps thresholds against the labeled validation set and
  plots **precision/recall and fairness per threshold** so an administrator
  picks an operating point *consciously* (e.g., favor recall-of-compliant to
  minimize false accusations). Persist the chosen point with a justification.

### Hot-reload + governance
- Policies are versioned (git + DB row). Changing a rule creates a new
  `policy_version`; every decision stores which version judged it - so an
  appeal can be evaluated against the rules in force at that moment.

---

## 9. JSON Rule Examples

`campus_profiles/default.json`:

```json
{
  "profile_id": "default",
  "version": "2026.06.01",
  "display_name": "Standard Campus Dress Code",
  "language": "en",
  "decision_bands": {
    "compliant":      { "min_compliance": 0.80, "max_uncertainty": 0.30 },
    "minor_violation":{ "min_compliance": 0.50, "max_uncertainty": 0.30 },
    "major_violation":{ "max_compliance": 0.50, "max_uncertainty": 0.30 },
    "insufficient_evidence": { "min_uncertainty": 0.30 }
  },
  "region_exposure_limits": {
    "shoulder": 0.40, "upper_arm": 0.60, "midriff": 0.15,
    "thigh": 0.20, "knee": 0.50, "calf": 1.00
  },
  "rules": [
    {
      "id": "UPPER_SLEEVELESS",
      "description": "Sleeveless / tank tops are not permitted",
      "category": "upper_body",
      "severity": 0.7,
      "weight": 1.0,
      "citation": "Student Handbook §4.2(a)",
      "when": {
        "any": [
          { "attr": "upper.sleeveless", "op": ">=", "value": 0.6 },
          { "attr": "upper.type", "op": "in", "value": ["tank_top", "sleeveless_top"] }
        ]
      }
    },
    {
      "id": "LOWER_ABOVE_KNEE",
      "description": "Hemline must reach the knee",
      "category": "lower_body",
      "severity": 0.8,
      "weight": 1.2,
      "citation": "Student Handbook §4.2(c)",
      "when": {
        "all": [
          { "attr": "lower.hemline_above_knee", "op": ">=", "value": 0.6 },
          { "attr": "exposure.thigh", "op": ">", "value": 0.20 }
        ]
      }
    },
    {
      "id": "MIDRIFF_CROP",
      "description": "Crop tops exposing the midriff are not permitted",
      "category": "upper_body",
      "severity": 0.8,
      "weight": 1.1,
      "citation": "Student Handbook §4.2(b)",
      "when": { "attr": "exposure.midriff", "op": ">", "value": 0.15 }
    },
    {
      "id": "FOOTWEAR_SANDALS",
      "description": "Open footwear discouraged in laboratory zones",
      "category": "footwear",
      "severity": 0.3,
      "weight": 0.5,
      "enabled_in_zones": ["lab", "workshop"],
      "citation": "Lab Safety Policy §2.1",
      "when": { "attr": "footwear.type", "op": "in", "value": ["sandals", "flip_flops"] }
    },
    {
      "id": "SHEER_ADVISORY",
      "description": "Possible sheer/transparent fabric - flag for human review only",
      "category": "advisory",
      "severity": 0.2,
      "weight": 0.0,
      "advisory_only": true,
      "citation": "Discretionary review",
      "when": { "attr": "upper.transparent_sheer", "op": ">", "value": 0.7 }
    }
  ]
}
```

`thresholds.json` (global, tunable):

```json
{
  "attribute_decision_threshold": 0.6,
  "evidence_quality_floor": 0.45,
  "temporal": { "ema_alpha": 0.4, "debounce_k_of_m": [3, 5] },
  "calibration": { "method": "temperature", "per_head": true }
}
```

---

## 10. Example API Responses

`POST /v1/infer` (single frame) or WebSocket stream → per-track decisions:

```json
{
  "frame_id": "cam03-2026-06-09T10:14:02.331Z-000812",
  "camera_id": "cam03",
  "profile_id": "default",
  "policy_version": "2026.06.01",
  "persons": [
    {
      "track_id": 47,
      "bbox": [412, 96, 233, 564],
      "decision": "major_violation",
      "scores": {
        "exposure_score": 0.71,
        "formality_score": 0.34,
        "compliance_score": 0.41,
        "uncertainty_score": 0.16,
        "overall_violation": 0.87
      },
      "evidence": {
        "upper": { "type": "tank_top", "conf": 0.91,
                   "attributes": { "sleeveless": 0.93, "transparent_sheer": 0.08 } },
        "lower": { "type": "shorts", "conf": 0.88,
                   "attributes": { "hemline_above_knee": 0.82 } },
        "footwear": { "type": "sneakers", "conf": 0.79 },
        "exposure": { "shoulder": 0.78, "upper_arm": 0.65,
                      "midriff": 0.04, "thigh": 0.33, "knee": 0.61 },
        "evidence_quality": 0.84
      },
      "matched_rules": [
        { "id": "UPPER_SLEEVELESS", "weight": 1.0, "severity": 0.7,
          "citation": "Student Handbook §4.2(a)" },
        { "id": "LOWER_ABOVE_KNEE", "weight": 1.2, "severity": 0.8,
          "citation": "Student Handbook §4.2(c)" }
      ],
      "explanation": "Detected sleeveless upper garment (tank top) with exposed shoulders (0.78 > limit 0.40) and shorts with hemline above the knee (thigh exposure 0.33 > limit 0.20). Two rules contributed. Campus policy violation score: 0.87 (HIGH). Evidence quality good (0.84).",
      "action": "alert_and_log"
    }
  ],
  "latency_ms": { "detect": 7, "parse": 19, "garment": 12, "attr": 9, "rules": 1, "total": 53 }
}
```

`insufficient_evidence` example (occlusion):

```json
{
  "track_id": 51,
  "decision": "insufficient_evidence",
  "scores": { "uncertainty_score": 0.62, "overall_violation": null },
  "evidence": { "evidence_quality": 0.31, "reason": "lower body occluded (0.7), low crop resolution" },
  "matched_rules": [],
  "explanation": "Lower body is mostly occluded; not enough reliable evidence to assess compliance. No alert raised; queued for optional human review.",
  "action": "review_optional"
}
```

### Core endpoints
```
POST   /v1/infer                 single image → decision
WS     /v1/stream                live frames ↔ per-track decisions
GET    /v1/profiles              list policy profiles
GET    /v1/profiles/{id}         fetch a profile (with version)
PUT    /v1/profiles/{id}         update policy (creates new version)
POST   /v1/profiles/{id}/tune    run threshold sweep, return PR/fairness curves
GET    /v1/events                query violation events (filters, pagination)
GET    /v1/events/{id}           full event + evidence + frame ref
POST   /v1/events/{id}/review    human reviewer overrides/confirms (audit)
GET    /v1/health  /v1/metrics   liveness + Prometheus metrics
```

---

## 11. Database Schema

PostgreSQL (operational) + object storage (frames/clips) + optional TimescaleDB
for high-rate metrics.

```sql
-- Cameras / zones
CREATE TABLE cameras (
  id            TEXT PRIMARY KEY,
  location      TEXT NOT NULL,
  zone          TEXT,                       -- lab, library, entrance...
  active_profile TEXT REFERENCES policy_versions(profile_id),
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- Policy versioning (immutable, append-only)
CREATE TABLE policy_versions (
  profile_id    TEXT NOT NULL,
  version       TEXT NOT NULL,
  document      JSONB NOT NULL,             -- the full profile JSON
  created_by    TEXT,
  created_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (profile_id, version)
);

-- Detection events (one per track-decision, deduped by debounce)
CREATE TABLE events (
  id              BIGSERIAL PRIMARY KEY,
  camera_id       TEXT REFERENCES cameras(id),
  track_id        INTEGER NOT NULL,
  profile_id      TEXT NOT NULL,
  policy_version  TEXT NOT NULL,
  decision        TEXT NOT NULL,            -- compliant|minor|major|insufficient
  exposure_score   REAL,
  formality_score  REAL,
  compliance_score REAL,
  uncertainty_score REAL,
  overall_violation REAL,
  evidence_quality REAL,
  explanation     TEXT,
  frame_ref       TEXT,                     -- object-store key (privacy-controlled)
  created_at      TIMESTAMPTZ DEFAULT now(),
  FOREIGN KEY (profile_id, policy_version)
      REFERENCES policy_versions(profile_id, version)
);
CREATE INDEX idx_events_cam_time ON events (camera_id, created_at DESC);
CREATE INDEX idx_events_decision ON events (decision);

-- Per-event structured evidence (attributes, rules) for audit/analytics
CREATE TABLE event_evidence (
  event_id     BIGINT REFERENCES events(id) ON DELETE CASCADE,
  payload      JSONB NOT NULL,              -- full evidence object + matched_rules
  PRIMARY KEY (event_id)
);

-- Human review / appeals (closes the loop, accountability)
CREATE TABLE reviews (
  id           BIGSERIAL PRIMARY KEY,
  event_id     BIGINT REFERENCES events(id),
  reviewer     TEXT NOT NULL,
  verdict      TEXT NOT NULL,               -- confirm|override_compliant|override_violation
  note         TEXT,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Active-learning queue (low-confidence frames for labeling)
CREATE TABLE label_queue (
  id           BIGSERIAL PRIMARY KEY,
  frame_ref    TEXT NOT NULL,
  reason       TEXT,                        -- low_conf | model_disagreement
  status       TEXT DEFAULT 'pending',
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Model registry (which weights/version produced a decision)
CREATE TABLE model_registry (
  id           TEXT PRIMARY KEY,            -- e.g. segformer-b2@campus-v3
  role         TEXT NOT NULL,               -- detector|parser|garment|attr...
  artifact_uri TEXT NOT NULL,
  metrics      JSONB,
  created_at   TIMESTAMPTZ DEFAULT now()
);
```

> **Privacy:** `frame_ref` images contain biometric data. Store encrypted,
> short retention (e.g., 7–30 days unless an appeal is open), access-logged,
> auto-purged. Prefer storing **evidence vectors over raw frames** where policy
> allows. See §14.

---

## 12. Evaluation Metrics

Evaluate **per stage** and **end-to-end**, and always **per fairness group**.

| Component | Metrics |
|---|---|
| Person detection | mAP@[.5:.95], AR, miss-rate under occlusion (CrowdHuman split) |
| Tracking | MOTA, IDF1, ID-switches, track fragmentation |
| Pose | PCK@0.2, OKS-AP, keypoint visibility accuracy (knee/shoulder specifically) |
| Human parsing | mIoU, per-class IoU (arms/legs/upper/lower), pixel acc |
| Garment detection | mAP, per-class AP, mask IoU |
| Attribute classification | per-attribute **F1, precision, recall, AP**; macro-F1 (handles imbalance); confusion per attribute |
| Exposure estimation | MAE of exposure ratio vs human-labeled; correlation; **per skin-tone bucket** |
| Calibration | **ECE**, MCE, reliability diagrams, Brier score (per head, pre/post temperature scaling) |
| **System (decision)** | precision/recall/F1 of `violation` vs human ground truth; **false-accusation rate** (compliant judged violation) - minimize this; confusion over 4 bands |
| Temporal | flicker rate (decision changes/sec on stable subject), debounce latency |
| Fairness (§14) | metric gaps across skin-tone / gender-presentation / body-type / garment-color groups; equal-opportunity difference |
| Robustness | metric degradation under lighting/occlusion/blur perturbation suites |
| Performance | end-to-end latency (p50/p95), FPS, GPU mem, throughput per stream |

**Headline metric to optimize:** **false-accusation rate at fixed coverage** -
because in a dress-code system, wrongly flagging a compliant student is the most
harmful error. Report it prominently and gate releases on it.

---

## 13. Deployment Strategy

### Topologies
1. **Edge (per-camera/kiosk):** NVIDIA **Jetson Orin** (NX/AGX). All models
   TensorRT-INT8. Best for privacy (frames never leave device) and latency.
2. **On-prem server:** 1–N RTX/A-series GPUs + **Triton** serving all models,
   handling many RTSP streams; campus keeps all data in-house (recommended for
   a student-facing system).
3. **Hybrid:** edge runs detection+tracking, streams crops to server for heavy
   parsing/attributes (bandwidth-friendly, central policy).

### Packaging & ops
- **Containers:** separate images for `perception` (CUDA/TensorRT), `api`
  (FastAPI/uvicorn), `ui`, `db`. **docker-compose** for single-node;
  **Kubernetes + Helm** for multi-node.
- **Serving:** Triton (dynamic batching, model ensembles), gRPC internal,
  REST/WebSocket external.
- **Observability:** Prometheus + Grafana (latency, FPS, GPU, queue depth,
  per-stage drop rate), structured logs, OpenTelemetry traces.
- **CI/CD:** model artifacts in registry; canary a new model in shadow mode,
  compare metrics + fairness vs incumbent before promotion.
- **Config:** all policy/threshold/model-registry via mounted config + hot
  reload; no redeploy to change a campus rule.

### Quantization strategy
| Model | Recommended precision | Notes |
|---|---|---|
| Person detector | INT8 (PTQ + calibration set) | big speedup, minimal mAP loss |
| Pose | FP16 (INT8 if validated) | keypoint precision sensitive |
| Parsing (SegFormer) | FP16 | INT8 can hurt thin-region IoU; QAT if needed |
| Garment det/seg | INT8 with QAT fallback | validate mask IoU after PTQ |
| CLIP attribute backbone | FP16 | keep embeddings stable; probes are tiny |

- **PTQ first** (post-training, with a representative calibration set drawn from
  campus footage), measure metric drop; if a head regresses, switch that head to
  **QAT** (quantization-aware training) or keep it FP16. Always re-check
  **calibration (ECE)** after quantizing - quantization can de-calibrate
  confidences.
- Prune/distill later (e.g., distill SegFormer-B2 → B0) for tighter edge budgets.

### Edge considerations
- Power/thermal budget on Jetson → cap heavy-stage cadence; dynamic resolution.
- Model warmup at boot (TensorRT engine build cached per device).
- Offline-first: local policy cache, store-and-forward events when network down.
- Tamper/audit: signed model artifacts, read-only policy in field, audit log.

---

## 14. Fairness, Reliability & Confidence Calibration

This system makes judgments about people's bodies and clothing. Treat fairness
and governance as **first-class requirements**, not an afterthought.

### Confidence calibration
- **Temperature scaling** per model head on a held-out set so a reported "0.87"
  is a *true* 87% - essential because the rule engine thresholds on these
  numbers and the four-score methodology multiplies them.
- Report **ECE + reliability diagrams** per head; re-calibrate after any
  quantization or domain shift.
- **Uncertainty-aware abstention:** the `insufficient_evidence` band is the
  product of calibration - the system declines rather than guesses, which is
  both fairer and more reliable under occlusion/lighting/pose (your stated
  weaknesses).

### Fairness guardrails (build these in)
- **No protected-attribute models.** Pipeline contains no gender/ethnicity/age
  classifier. Decisions rest only on garment geometry + policy.
- **Skin-tone-invariant exposure.** Geometry-based (parsing + pose), validated
  per Fitzpatrick bucket; gate release on the *gap* between buckets, not just
  the mean.
- **Headwear/religious dress safe-listed.** Hijab, turban, etc. must never be
  flagged; include them abundantly in data and add explicit allow-rules.
- **Bias dashboard:** per-group precision/recall/false-accusation continuously
  monitored in production; alert on drift.

### Governance / ethics (academically + operationally essential)
- **Human-in-the-loop for consequences.** Clothic AI outputs *advisory* flags;
  a human makes any disciplinary decision. The system assists, it does not
  punish. The `reviews` table enforces this.
- **Contestability.** Every decision ships its evidence + handbook citation, so
  a student can appeal against specific measurements and the rules in force.
- **Privacy & data minimization.** Prefer storing evidence vectors over raw
  frames; encrypt, short-retain, access-log, auto-purge; publish a retention
  policy. Get ethics/IRB approval for the student-data capture.
- **Transparency.** Signage that the system is active; publish the policy JSON
  and the appeal process. A dress-code monitor that students can read and
  challenge is far more legitimate than a black box.

> Building it this way (attributes + transparent policy + human review + audit)
> is not just nicer - it's what makes the project **defensible as a thesis** and
> deployable without becoming an unaccountable surveillance tool.

---

## 15. Folder Structure

```
clothic-ai/
├── README.md
├── pyproject.toml                # deps, tooling (ruff, mypy, pytest)
├── docker-compose.yml
├── .dvc/                         # dataset/model version control
├── configs/
│   ├── model_registry.yaml
│   ├── thresholds.json
│   └── pipeline.yaml             # stage cadence, batch sizes, devices
├── campus_profiles/
│   ├── default.json
│   ├── lab_safety.json
│   └── exam_formal.json
├── ontology/
│   ├── taxonomy.yaml
│   └── mappings/                 # per-dataset label maps
├── data/                         # DVC-tracked (not in git)
│   ├── raw/                      # source datasets
│   ├── unified/                  # normalized COCO+attributes
│   └── campus/                   # in-situ captures (consented)
├── src/clothic/
│   ├── ingest/                   # capture, decode, preprocess (S1)
│   ├── perception/
│   │   ├── detect/               # person detection (S2)
│   │   ├── track/                # ByteTrack (S3)
│   │   ├── pose/                 # RTMPose (S4)
│   │   ├── parsing/              # SegFormer (S5)
│   │   ├── garment/              # YOLOv11-seg / RT-DETR (S6)
│   │   ├── attributes/           # CLIP probes (S7)
│   │   └── exposure/             # geometric exposure (S8)
│   ├── fusion/                   # confidence fusion + temporal (S9)
│   ├── reasoning/
│   │   ├── rule_engine.py        # predicate evaluation (S10)
│   │   ├── scoring.py            # four-score aggregation (S11)
│   │   └── calibration.py        # temperature scaling
│   ├── explain/                  # NL templates + overlay (S12)
│   ├── api/                      # FastAPI + WebSocket (S13)
│   ├── ui/                       # dashboard (or separate frontend/)
│   ├── persistence/              # DB models, repositories
│   ├── pipeline.py               # orchestrates stages, queues
│   └── schemas.py                # typed contracts between stages
├── training/
│   ├── stage_b_parsing/
│   ├── stage_c_garment/
│   ├── stage_d_attributes/
│   ├── augment.py
│   └── datamodule.py
├── tools/
│   ├── dataset_unifier.py
│   ├── label_cleaning.py
│   ├── threshold_tuner.py
│   └── fairness_audit.py
├── eval/                         # metric suites, perturbation tests
├── deploy/
│   ├── triton/                   # model repo, ensemble config
│   ├── tensorrt/                 # engine build scripts
│   ├── jetson/                   # edge build
│   └── k8s/                      # helm charts
└── tests/                        # unit + integration + golden-file
```

---

## 16. Future Improvements Roadmap

**Phase 0 - Foundation (now → MVP)**
- Unified ontology + dataset_unifier; assemble Fashionpedia + DeepFashion2 + LIP.
- Stand up the staged pipeline with pretrained models (no fine-tune yet).
- Rule engine + four-score + templated explanations + basic API.
- Shadow-mode logging on real cameras (no alarms) to gather data.

**Phase 1 - Campus adaptation**
- Consented capture; fine-tune garment + parsing + attribute heads on campus data.
- Temperature calibration; threshold tuning with fairness curves.
- Temporal fusion + debounce; ByteTrack integration; per-track alerts.
- Dashboard UI with evidence overlay + appeal workflow.

**Phase 2 - Hardening & fairness**
- Full fairness audit across skin-tone/gender/body-type buckets; close gaps.
- Robustness suites (lighting/occlusion/blur); active-learning loop in prod.
- TensorRT/INT8 + Triton; meet latency/throughput SLOs.
- Privacy: retention automation, encryption, access logs, ethics sign-off.

**Phase 3 - Scale & intelligence**
- Multi-camera, multi-zone, time-based profiles; edge (Jetson) rollout.
- Open-vocabulary rule authoring: admin types a new rule in natural language →
  GroundingDINO/CLIP bootstraps detection → human validates → activated.
- Multi-task distilled backbone for lower edge cost.
- Self-monitoring drift detection; auto-flag when a model needs recalibration.

**Phase 4 - Research extensions**
- Causal/counterfactual explanations ("would be compliant if sleeves reached
  the elbow").
- VLM (e.g., a vision-language model) as an *auditor* that double-checks
  low-confidence decisions in natural language - as a second opinion feeding
  human review, never as the sole judge.
- Federated learning across campuses without sharing raw images.

---

### One-paragraph summary
Clothic AI replaces a single YOLO "modest/not" classifier with a modular pipeline
that **measures observable garment attributes** (type, sleeve, hemline-vs-knee,
rips, sheerness proxies, and skin-exposure ratios derived from parsing + pose)
and feeds them into a **transparent, JSON-configurable rule engine** that
produces four calibrated sub-scores and a written, citable explanation. The
neural nets never learn the value judgment; policy does - making the system
explainable, contestable, per-campus configurable, robust via temporal fusion
and confidence-gated abstention, fair via geometry-based (not skin-color-based)
exposure, and deployable from cloud to Jetson edge with TensorRT/INT8.
```