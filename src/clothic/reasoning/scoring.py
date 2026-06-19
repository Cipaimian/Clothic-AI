"""Four-score aggregation + decision banding.

Replaces the old binary "sopan / tidak sopan" with four interpretable numbers:

* ``exposure_score``    -- how far body-region exposure exceeds policy limits.
* ``formality_score``   -- how formal the outfit reads (garments + footwear).
* ``compliance_score``  -- 1 minus the saturated weighted rule severity.
* ``uncertainty_score`` -- 1 minus the evidence quality of the observation.
* ``overall_violation`` -- the headline violation magnitude (None if uncertain).

All constants live in ``configs/thresholds.json`` so an administrator can tune
the operating point consciously (see ``tools/threshold_tuner``).
"""

from __future__ import annotations

import math
from typing import Any

from clothic.schemas import Decision, MatchedRule, PersonObservation, Scores

# Defaults; overridden by configs/thresholds.json -> "scoring".
_DEFAULTS = {
    "violation_saturation_k": 1.25,  # higher => rules saturate violation faster
    "formality_baseline": 0.5,
}


class ScoringEngine:
    def __init__(self, profile: dict[str, Any], thresholds: dict[str, Any] | None = None):
        self.profile = profile
        thresholds = thresholds or {}
        self.cfg = {**_DEFAULTS, **thresholds.get("scoring", {})}
        self.bands = profile.get("decision_bands", {})
        self.exposure_limits: dict[str, float] = profile.get("region_exposure_limits", {})
        self.formality_cfg: dict[str, Any] = profile.get("formality", {})

    # -- individual scores -------------------------------------------------

    def exposure_score(self, obs: PersonObservation) -> float:
        """Aggregate how much each region's exposure exceeds its policy limit.

        For each region, ``excess = max(0, exposure - limit) / (1 - limit)``
        normalises the overshoot into 0..1; the score is the max overshoot so a
        single egregious region is not diluted by compliant ones.
        """
        if not obs.exposure:
            return 0.0
        overshoots = []
        for region, value in obs.exposure.items():
            limit = self.exposure_limits.get(region, 1.0)
            denom = max(1e-6, 1.0 - limit)
            overshoots.append(max(0.0, value - limit) / denom)
        return min(1.0, max(overshoots)) if overshoots else 0.0

    def formality_score(self, obs: PersonObservation) -> float:
        """Heuristic formality from garment 'formal' attributes + footwear."""
        base = float(self.formality_cfg.get("baseline", self.cfg["formality_baseline"]))
        score = base
        formal_footwear = set(self.formality_cfg.get("formal_footwear", ["formal_shoes", "shoes_closed"]))
        casual_footwear = set(self.formality_cfg.get("casual_footwear", ["sandals", "flip_flops"]))

        for slot in ("upper", "lower"):
            garment = getattr(obs, slot)
            if garment is not None:
                score += 0.25 * garment.attr("formal_style")
                score -= 0.15 * garment.attr("ripped_torn")
        if obs.footwear is not None:
            if obs.footwear.type in formal_footwear:
                score += 0.15
            elif obs.footwear.type in casual_footwear:
                score -= 0.15
        return float(min(1.0, max(0.0, score)))

    def violation_magnitude(self, matched: list[MatchedRule]) -> float:
        """Saturating map from summed weighted severity to 0..1."""
        raw = sum(r.weight * r.severity for r in matched if not r.advisory_only)
        k = float(self.cfg["violation_saturation_k"])
        return float(1.0 - math.exp(-k * raw))

    # -- combination + banding --------------------------------------------

    def score(self, obs: PersonObservation, matched: list[MatchedRule]) -> tuple[Scores, Decision]:
        uncertainty = float(min(1.0, max(0.0, 1.0 - obs.evidence_quality)))
        rule_violation = self.violation_magnitude(matched)
        exposure_excess = self.exposure_score(obs)
        # Pixel-measured exposure over the profile's region limits is a
        # first-class violation signal, on equal footing with fired rules: the
        # headline verdict must reflect what was actually measured, not only the
        # (softer) garment-attribute rules. Combined as a probabilistic OR so
        # either signal alone can drive the verdict and the result stays in 0..1.
        overall = 1.0 - (1.0 - rule_violation) * (1.0 - exposure_excess)
        compliance = 1.0 - overall

        decision = self._band(compliance, uncertainty)
        # When evidence is insufficient we withhold the violation magnitude so
        # downstream consumers never treat an uncertain frame as a hard verdict.
        overall_out: float | None = None if decision == Decision.INSUFFICIENT_EVIDENCE else overall

        scores = Scores(
            exposure_score=self.exposure_score(obs),
            formality_score=self.formality_score(obs),
            compliance_score=compliance,
            uncertainty_score=uncertainty,
            overall_violation=overall_out,
        )
        return scores, decision

    def _band(self, compliance: float, uncertainty: float) -> Decision:
        b = self.bands
        insuff = b.get("insufficient_evidence", {})
        if uncertainty > insuff.get("min_uncertainty", 0.30):
            return Decision.INSUFFICIENT_EVIDENCE

        comp = b.get("compliant", {})
        if compliance >= comp.get("min_compliance", 0.80):
            return Decision.COMPLIANT

        major = b.get("major_violation", {})
        if compliance < major.get("max_compliance", 0.50):
            return Decision.MAJOR_VIOLATION

        return Decision.MINOR_VIOLATION
