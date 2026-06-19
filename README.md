<div align="center">

# Clothic AI

### Explainable, real-time campus outfit-compliance

A **Visual Attribute Recognition + Policy Reasoning Engine**: it detects what a
person is *wearing*, measures exposure geometrically, and lets a transparent,
editable rule engine decide compliance, never the neural net.

`MIT licensed` · `Python 3.10+` · `FastAPI` · `60 tests passing`

</div>

---

Instead of a single model that emits a binary "sopan / tidak sopan" label,
Clothic AI detects **observable visual attributes** of what a person is wearing
and feeds them into a **transparent, JSON-configurable rule engine** that
produces four calibrated scores and a written, citable explanation.
**The models never learn the value judgement, policy does.**

```
Image -> Human Parsing -> Clothing Attributes -> Exposure Estimation
      -> Policy Reasoning -> Explainable Decision
```

## Why this design

- **Explainable.** Every verdict ships its evidence, the rules that fired, and a
  handbook citation, so a student can contest it.
- **Configurable.** Campus rules and thresholds live in editable JSON. Change
  policy with zero retraining and zero code changes.
- **Fair and reliable.** Exposure is computed geometrically (parsing + pose),
  not from skin colour. Under occlusion or blur the system **abstains**
  (`insufficient_evidence`) instead of risking a false accusation.
- **Modular.** Each stage is a swappable component behind a typed contract.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
architecture, model and dataset recommendations, and roadmap.

## Quick start

The reasoning core, CLI, and API run with **no heavy ML dependencies** thanks to
a built-in mock perception backend.

```bash
pip install -e .            # base: numpy, pydantic, pyyaml
python -m pytest            # 60 tests pass (+1 skipped)

# Run the pipeline on scripted personas (exercises every decision band):
python -m clothic.cli demo --profile default --frames 5
python -m clothic.cli profiles
python -m clothic.cli explain --profile default      # one full verdict as JSON
```

### Run the API only

```bash
pip install -e ".[api]"
uvicorn clothic.api.app:app --reload
# GET  /v1/health   /v1/profiles   /v1/profiles/{id}
# POST /v1/infer    (mock)         /v1/infer_image (real backend)
```

## Run the web app (UI + API)

The API serves the web frontend (`web/`) itself via `StaticFiles`, so a **single
process** serves BOTH the SPA (at `/`) and the API (`/v1/*`) on the same origin.
One terminal, no CORS, and no `API_BASE` host to configure (the frontend calls
the API with relative paths).

```bash
# macOS / Linux
CLOTHIC_BACKEND=mock uvicorn clothic.api.app:app --host 127.0.0.1 --port 8000
```

```powershell
# Windows PowerShell
$env:CLOTHIC_BACKEND = "mock"
python -m uvicorn clothic.api.app:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`, then open **http://127.0.0.1:8000**. The
SPA loads at `/` and the API lives under `/v1/*`. The **webcam** mode needs a
secure context, which `http://127.0.0.1` satisfies, so it works out of the box;
upload mode works regardless.

> Health check: http://127.0.0.1:8000/v1/health returns `{"status":"ok",...}` and
> the navbar status turns green. Hard-refresh (Ctrl+Shift+R) after editing CSS/JS.

### Backend modes (`CLOTHIC_BACKEND`)

Override the default selected in `configs/pipeline.yaml`:

| Mode | Speed (CPU) | Notes |
|------|-------------|-------|
| `mock` | instant | synthetic personas, best for UI / layout work, no ML deps |
| `ultralytics` | ~2 s/photo | real YOLO detection, but often **abstains** (no pixel exposure) |
| `full` | slow | Sapiens parsing, most accurate; upload-only, too slow for live webcam |

Real modes need the heavy extras:

```bash
pip install -e ".[vision,perception]"   # ultralytics + torch + opencv + transformers
python scripts/demo_webcam.py --profile default --source 0   # CLI webcam demo
```

The `full` backend runs person-detect -> pose -> Sapiens parsing -> garment ->
FashionSigLIP, producing **pixel-measured** exposure. The `ultralytics` backend
is detection-only (parsing and pose disabled), so exposure is attribute-derived
and `evidence_quality` is discounted accordingly.

## Architecture at a glance

```
frame
  -> perception (person detect -> track -> [parse/pose] -> garment -> attributes -> exposure)
  -> temporal fusion (EMA smoothing + K-of-M debounce, per track)
  -> rule engine (JSON policy predicates)
  -> four-score (exposure / formality / compliance / uncertainty)
  -> explanation (deterministic templated NL + citations)
  -> FrameResult (API / UI / audio / DB)
```

### The four scores (replacing binary sopan/tidak)

| Score | Meaning |
|-------|---------|
| `exposure_score` | how far body-region exposure exceeds policy limits |
| `formality_score` | how formal the outfit reads |
| `compliance_score` | 1 minus saturated weighted rule severity |
| `uncertainty_score` | 1 minus evidence quality (drives abstention) |
| `overall_violation` | headline magnitude (`null` when uncertain) |

## Project layout

```
campus_profiles/   editable JSON policy profiles (default, lab_safety, exam_formal)
configs/           thresholds, pipeline, model registry
ontology/          canonical taxonomy + per-dataset label mappings
src/clothic/
  schemas.py       typed stage contracts (pydantic)
  perception/      backends (mock + ultralytics) + geometric exposure
  fusion/          temporal smoothing + debounce
  reasoning/       rule_engine.py + scoring.py
  explain/         templated explanation generator
  pipeline.py      orchestrator
  api/  cli.py     FastAPI service + CLI
web/               static frontend SPA (index.html, style.css, app.js)
scripts/           demo_webcam.py + verify_full_backend.py
tests/             pytest suite (rule engine, scoring, pipeline, fusion)
deploy/db/         SQLite schema
docs/              full redesign specification
```

## Editing policy

Add or change a rule in `campus_profiles/*.json`, no code change needed:

```json
{
  "id": "UPPER_SLEEVELESS",
  "description": "Sleeveless / tank tops are not permitted",
  "severity": 0.7, "weight": 1.0,
  "citation": "Student Handbook 4.2(a)",
  "when": { "attr": "upper.sleeveless", "op": ">=", "value": 0.6 }
}
```

Predicate grammar: `all` / `any` / `not` / leaf `{attr, op, value}` with operators
`> >= < <= == != in not_in exists` over dotted attribute paths
(`upper.sleeveless`, `exposure.thigh`, `footwear.type`, ...).

## Ethics and governance

Clothic AI outputs **advisory** flags with evidence; a human makes any
disciplinary decision (`reviews` table). No protected-attribute
(gender / ethnicity) models are used. Store evidence vectors over raw frames
where possible; encrypt, short-retain, auto-purge. See section 14 of the
redesign doc.

## License

Released under the [MIT License](LICENSE).
