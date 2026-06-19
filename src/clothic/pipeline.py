"""End-to-end Clothic AI pipeline orchestrator.

Wires the stages together:

    frame --> perception --> temporal smoothing --> rule engine
          --> four-score --> debounce --> explanation --> FrameResult

The orchestrator is deliberately thin: every real decision lives in a tested,
single-responsibility module. Swap the perception backend or edit the policy
JSON and nothing else changes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from clothic.config import load_calibration, load_profile, load_thresholds
from clothic.explain.counterfactual import CounterfactualEngine
from clothic.explain.explainer import Explainer
from clothic.fusion.temporal import TemporalFuser
from clothic.perception import get_backend
from clothic.perception.base import PerceptionBackend
from clothic.reasoning.calibration import Calibrator, calibrate_observation
from clothic.reasoning.rule_engine import RuleEngine
from clothic.reasoning.scoring import ScoringEngine
from clothic.schemas import Decision, FrameResult


class ClothicPipeline:
    def __init__(
        self,
        profile_id: str = "default",
        backend: PerceptionBackend | str = "mock",
        camera_id: str = "cam0",
        zone: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        enable_temporal: bool = True,
    ):
        # Temporal smoothing + debounce suit video streams. For single-image
        # inference set enable_temporal=False so a lone frame isn't suppressed.
        self.enable_temporal = enable_temporal
        self.profile = load_profile(profile_id)
        thresholds = load_thresholds()

        self.camera_id = camera_id
        self.zone = zone
        self.rule_engine = RuleEngine(self.profile)
        self.scoring = ScoringEngine(self.profile, thresholds)
        self.explainer = Explainer(self.profile.get("region_exposure_limits", {}))
        self.counterfactual = CounterfactualEngine(self.rule_engine, self.scoring)
        self.calibrator = Calibrator(load_calibration())

        temporal = thresholds.get("temporal", {})
        k, m = temporal.get("debounce_k_of_m", [3, 5])
        self.fuser = TemporalFuser(alpha=temporal.get("ema_alpha", 0.4), k=k, m=m)

        if isinstance(backend, str):
            self.backend: PerceptionBackend = get_backend(backend, **(backend_kwargs or {}))
        else:
            self.backend = backend

        self._frame_counter = 0

    def process_frame(self, frame: Any = None) -> FrameResult:
        t0 = time.perf_counter()
        observations = self.backend.observe(frame, frame_index=self._frame_counter)
        t_perc = time.perf_counter()

        persons = []
        for obs in observations:
            obs = calibrate_observation(obs, self.calibrator)
            fused = self.fuser.smooth(obs) if self.enable_temporal else obs
            matched = self.rule_engine.match(fused, zone=self.zone)
            scores, decision = self.scoring.score(fused, matched)
            if self.enable_temporal:
                decision = self.fuser.debounce(fused.track_id, decision)
            remediation = None
            if decision in (Decision.MINOR_VIOLATION, Decision.MAJOR_VIOLATION):
                remediation = self.counterfactual.generate(fused, matched, zone=self.zone)
            persons.append(
                self.explainer.build_decision(fused, scores, decision, matched, remediation)
            )
        t_reason = time.perf_counter()

        frame_id = (
            f"{self.camera_id}-{datetime.now(timezone.utc).isoformat()}-"
            f"{self._frame_counter:06d}"
        )
        result = FrameResult(
            frame_id=frame_id,
            camera_id=self.camera_id,
            profile_id=self.rule_engine.profile_id,
            policy_version=self.rule_engine.version,
            persons=persons,
            latency_ms={
                "perception": round((t_perc - t0) * 1000, 2),
                "reasoning": round((t_reason - t_perc) * 1000, 2),
                "total": round((t_reason - t0) * 1000, 2),
            },
        )
        self._frame_counter += 1
        return result

    def close(self) -> None:
        self.backend.close()
