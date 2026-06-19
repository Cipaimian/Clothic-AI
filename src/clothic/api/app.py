"""FastAPI application exposing Clothic AI.

Endpoints mirror the redesign spec. The default perception backend is the mock
backend, so the API runs and returns realistic decisions without any heavy
model dependencies. Point ``CLOTHIC_BACKEND=ultralytics`` to use real detection.

Run::

    pip install "clothic[api]"
    uvicorn clothic.api.app:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clothic.api.dashboard import DASHBOARD_HTML
from clothic.config import list_profiles, load_profile, resolve_backend_config
from clothic.persistence import EventStore
from clothic.pipeline import ClothicPipeline
from clothic.schemas import FrameResult

app = FastAPI(
    title="Clothic AI API",
    version="0.1.0",
    description="Explainable campus outfit-compliance: visual attributes + policy reasoning.",
)

# The Clothic AI web frontend runs from a separate local origin (file:// or a
# static dev server), so it needs CORS to reach this API. Stays fully local --
# this only relaxes the browser same-origin check, it opens no remote network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CLOTHIC_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Which perception backend + model weights run is decided by configs/pipeline.yaml
# (overridable with CLOTHIC_BACKEND). This is the single source of truth, so the
# API and the verify/CLI scripts all load the same models from the same config.
_BACKEND, _BACKEND_KWARGS = resolve_backend_config()
# Local SQLite store (privacy: stores evidence vectors, not raw frames).
_DB_PATH = os.environ.get("CLOTHIC_DB", "data/clothic.db")
_store = EventStore(_DB_PATH)


@lru_cache(maxsize=16)
def _pipeline(profile_id: str, camera_id: str, zone: Optional[str],
              enable_temporal: bool = True) -> ClothicPipeline:
    return ClothicPipeline(
        profile_id=profile_id, backend=_BACKEND, camera_id=camera_id, zone=zone,
        backend_kwargs=_BACKEND_KWARGS, enable_temporal=enable_temporal,
    )


class InferRequest(BaseModel):
    profile_id: str = "default"
    camera_id: str = "cam0"
    zone: Optional[str] = None


@app.get("/v1/health")
def health() -> dict:
    # Echo the configured model artifacts so operators can confirm, at a glance,
    # exactly which weights are judging frames.
    models = {k: v for k, v in _BACKEND_KWARGS.items()
              if k in ("person_weights", "garment_weights", "pose_weights",
                       "sapiens_checkpoint", "clip_model", "parser_type")}
    return {"status": "ok", "backend": _BACKEND, "models": models,
            "profiles": list_profiles()}


@app.get("/v1/profiles")
def get_profiles() -> dict:
    return {"profiles": list_profiles()}


@app.get("/v1/profiles/{profile_id}")
def get_profile(profile_id: str) -> dict:
    try:
        return load_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")


@app.post("/v1/infer", response_model=FrameResult)
def infer(req: InferRequest) -> FrameResult:
    """Run one inference step and persist the resulting decisions.

    With the mock backend this returns scripted personas (handy for demos and
    integration tests). With a real backend, send frames via ``/v1/infer_image``.
    """
    pipe = _pipeline(req.profile_id, req.camera_id, req.zone)
    _store.record_policy_version(pipe.profile)
    result = pipe.process_frame(None)
    _store.log_frame(result)
    return result


class ReviewRequest(BaseModel):
    reviewer: str
    verdict: str  # confirm | override_compliant | override_violation
    note: str = ""


@app.get("/v1/events")
def list_events(decision: Optional[str] = None, camera_id: Optional[str] = None,
                limit: int = 100) -> dict:
    return {"events": _store.query_events(decision=decision, camera_id=camera_id, limit=limit),
            "stats": _store.stats()}


@app.get("/v1/events/{event_id}")
def get_event(event_id: int) -> dict:
    event = _store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return event


@app.post("/v1/events/{event_id}/review")
def review_event(event_id: int, req: ReviewRequest) -> dict:
    """Record a human reviewer's verdict (the human-in-the-loop / appeal step)."""
    try:
        review_id = _store.add_review(event_id, req.reviewer, req.verdict, req.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"review_id": review_id, "event_id": event_id, "verdict": req.verdict}


@app.get("/ops", response_class=HTMLResponse)
def dashboard() -> str:
    """Minimal operator dashboard: live decisions, evidence, and appeals."""
    return DASHBOARD_HTML


@app.post("/v1/infer_image", response_model=FrameResult)
async def infer_image(
    file: UploadFile,
    profile_id: str = "default",
    camera_id: str = "cam0",
    zone: Optional[str] = None,
) -> FrameResult:
    """Run inference on an uploaded image (real backend)."""
    try:
        import cv2  # local import: only needed on the image path
    except ImportError:
        raise HTTPException(status_code=501, detail="Install clothic[vision] for image inference")

    data = await file.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    # Single image => no temporal debounce (a lone frame must not be suppressed).
    pipe = _pipeline(profile_id, camera_id, zone, enable_temporal=False)
    result = pipe.process_frame(arr)
    _store.log_frame(result)
    return result


# Serve the web frontend (web/) from the API itself, so a single
# `uvicorn clothic.api.app:app` serves BOTH the SPA (at /) and the API (/v1/*).
# Mounted LAST so the explicit API routes above always take precedence. Running
# same-origin means the frontend needs no CORS and no hard-coded API host (its
# API_BASE is relative) -- which also makes LAN/phone/remote deploys just work.
_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
