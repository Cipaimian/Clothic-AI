# Clothic AI - Project Status

_Last updated: 2026-06-17 · **60 tests passing (+1 skip)** · CPU-only · full
hybrid backend (Sapiens) re-verified end-to-end on real images with the new
13-class garment detector · **Clothic AI** (package `clothic`, env `CLOTHIC_*`)_

Legend: ✅ done & tested · 🟡 built but needs heavy deps to run · ⬜ not started

---

## ✅ Done (built + tested, CPU-only)

### Reasoning core
- ✅ Typed contracts (`schemas.py`) incl. `Remediation`.
- ✅ Rule engine (`reasoning/rule_engine.py`) - predicates, dotted paths, zones.
- ✅ Four-score + banding (`reasoning/scoring.py`) - reproduces the 0.87 example.
- ✅ Confidence calibration (`reasoning/calibration.py`) - temperature scaling +
  ECE, **wired live into the pipeline** (`calibrate_observation`, identity by
  default; `configs/calibration.json` overrides).

### Explainability
- ✅ Templated NL explanations (`explain/explainer.py`) with handbook citations.
- ✅ **Verified counterfactuals** (`explain/counterfactual.py`) - "to become
  compliant: …", re-checked against the engine; attached to every violation.

### Perception
- ✅ Mock backend + geometric exposure + evidence-quality model.

### Pipeline, fusion, interfaces
- ✅ Temporal fusion (EMA + K-of-M debounce).
- ✅ Orchestrator (`pipeline.py`): perception → calibrate → fuse → rules → score
  → debounce → counterfactual → explain.
- ✅ REST/WS API (`api/app.py`): health, profiles, infer, infer_image, **events,
  event detail, review/appeal** - all logging to SQLite.
- ✅ **Operator dashboard** (`api/dashboard.py`, served at `/`): live cards with
  scores, evidence, counterfactual, and an appeal button.
- ✅ CLI (`cli.py`) + webcam demo.

### Persistence & evaluation
- ✅ **SQLite event store** (`persistence/store.py`) - event logging, human
  review/appeals, label queue, retention purge, stats. Thread-safe for the API.
- ✅ **Evaluation harness** (`eval/metrics.py`) - PRF, macro-F1, false-accusation
  rate, **fairness gap across groups**.

### Tools (CPU)
- ✅ `tools/dataset_unifier.py` (tested: 388 imgs, 0 unmapped).
- ✅ `tools/fit_calibration.py` (fit per-head temperatures → config).
- ✅ `tools/threshold_tuner.py` (PR + fairness sweep).
- ✅ `tools/fairness_audit.py` (per-group metrics + gate).
- ✅ `tools/active_learning.py` (harvest uncertain events → label queue).

### Config, data, ops, docs
- ✅ 3 campus profiles, configs, ontology (+ Roboflow mapping).
- ✅ DB schema, full redesign doc, this status, legacy archive.

---

## ✅ Real perception stack - installed and VERIFIED end-to-end
Heavy deps are installed globally (torch 2.12+cpu, ultralytics 8.4, transformers
5.10, open_clip, cv2) and everything still **imports cleanly without them**
(`tests/test_lazy_imports.py`). Checkpoints on disk: `yolo11n.pt`,
`yolo11n-pose.pt`, legacy garment `best.pt`, and Sapiens 0.3B
(`models/sapiens/sapiens_0.3b_goliath.pt2`).

- ✅ **Ultralytics backend** (`perception/ultralytics_backend.py`) - YOLO person
  detect + ByteTrack + garment class-map (legacy `best.pt`); verified
  (singlet+shorts → major_violation 0.87).
- ✅ **Pose** (`perception/pose.py`) - knee/shoulder anchor lines for exposure.
- ✅ **Human parsing** (`perception/parsing.py`) - **Sapiens** (Goliath 28-class)
  → pixel-measured skin-exposure per region; SegFormer fallback.
- ✅ **CLIP attributes** (`perception/attributes.py`) - FashionSigLIP probes.
- ✅ **Full hybrid backend** (`perception/full_backend.py`) - person → pose →
  Sapiens parsing → garment → FashionSigLIP. **Verified end-to-end on real
  dataset images** (`scripts/verify_full_backend.py`): singlet+shorts →
  MAJOR_VIOLATION 0.87 (upper_arm exposure 0.90, fires SLEEVELESS+ABOVE_KNEE,
  verified counterfactual); tshirt+long-pants → COMPLIANT 0.00 (thigh/knee/calf
  exposure ~0). Exposure is now **pixel-measured** (quality no longer discounted
  for that reason). Wired into `configs/pipeline.yaml` (`backend: full`). Sapiens
  0.3B on CPU ≈ 32 s/frame - needs GPU/TensorRT for real-time (see deploy).

### Garment detector - trained & wired (2026-06-17)
- ✅ **13-class garment model trained** (`training/stage_c_garment/`, run
  `stage_c-7`) on the merged multi-source set (campus dresscode + DeepFashion2 +
  Fashionpedia). **mAP50-95 0.428 / mAP50 0.628** on the campus val split.
  Per-class highlights: dress 0.766, pants 0.663, skirt 0.500, shoe 0.427;
  weakest = sleeveless 0.168, id 0.318. Wired live as `garment_weights`
  (`configs/pipeline.yaml` → `runs/detect/runs_clothic/garment/stage_c-7/weights/best.pt`),
  replacing the legacy 4-class `best.pt`. Full backend re-verified on both
  fixtures (MAJOR_VIOLATION, correct rules + verified counterfactuals).

## 🟡 Built, needs heavy deps to actually run (lazy-imported; import-tested)
- 🟡 **Deployment** (`deploy/`) - TensorRT export, Triton repo, quantization guide.

---

## ⬜ Not started / future
- ⬜ Stage-B parsing fine-tune script (LIP/ATR/CIHP) + stage-D attribute probe
  training (currently zero-shot prompts).
- ⬜ Multi-camera/zone orchestration + time-based profile switching.
- ⬜ Open-vocabulary rule authoring (GroundingDINO bootstrap).
- ⬜ Prometheus/Grafana wiring; shadow-mode canary automation.
- ⬜ `git init` (local only) - **held pending your explicit OK** (you asked to
  keep everything isolated/local; a repo has no remote and can't push, but
  initialising version control is your call).

---

## Run it now
```bash
python -m pytest                                   # 60 pass (+1 skip)
PYTHONPATH=src python -m clothic.cli demo --frames 5            # mock backend
PYTHONPATH=src python scripts/verify_full_backend.py          # full stack, real images
pip install -e ".[api]" && uvicorn clothic.api.app:app --reload   # then open http://127.0.0.1:8000/
```
