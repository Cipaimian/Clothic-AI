"""Deterministic, template-based explanation generator.

Because Clothic AI' decision is already symbolic (a rule trace over measured
attributes), the honest explanation is literally that trace rendered to text.
No learned model invents justifications, so explanations can never disagree
with the decision. An LLM may *optionally* rephrase for tone in reports, but
must never add facts -- that integration is intentionally left out of the
core path.
"""

from __future__ import annotations

from clothic.explain.counterfactual import remediation_sentence
from clothic.perception.coverage import coverage_sentence, garment_coverage
from clothic.schemas import (
    Decision,
    MatchedRule,
    PersonDecision,
    PersonObservation,
    Remediation,
    Scores,
)

# Human-readable fragments per body region, used when an exposure limit is
# exceeded. Kept here (not in policy) because they describe physical facts.
_REGION_PHRASES = {
    "shoulder": "exposed shoulders",
    "upper_arm": "exposed upper arms",
    "midriff": "an exposed midriff",
    "thigh": "exposed thighs",
    "knee": "exposed knees",
    "calf": "exposed calves",
}

_SEVERITY_WORD = {
    Decision.COMPLIANT: "NONE",
    Decision.MINOR_VIOLATION: "LOW",
    Decision.MAJOR_VIOLATION: "HIGH",
    Decision.INSUFFICIENT_EVIDENCE: "UNKNOWN",
}


def choose_action(decision: Decision) -> str:
    return {
        Decision.COMPLIANT: "log",
        Decision.MINOR_VIOLATION: "notify",
        Decision.MAJOR_VIOLATION: "alert_and_log",
        Decision.INSUFFICIENT_EVIDENCE: "review_optional",
    }[decision]


class Explainer:
    def __init__(self, exposure_limits: dict[str, float] | None = None):
        self.exposure_limits = exposure_limits or {}

    def _garment_phrase(self, obs: PersonObservation) -> str:
        parts: list[str] = []
        if obs.upper is not None:
            sleeve = ""
            if obs.upper.attr("sleeveless") >= 0.6:
                sleeve = "sleeveless "
            parts.append(f"{sleeve}upper garment ({obs.upper.type}, conf {obs.upper.conf:.2f})")
        if obs.lower is not None:
            hem = ""
            if obs.lower.attr("hemline_above_knee") >= 0.6:
                hem = " with hemline above the knee"
            parts.append(f"{obs.lower.type}{hem}")
        if obs.footwear is not None:
            parts.append(f"{obs.footwear.type} footwear")
        if not parts:
            return "no clearly identifiable garments"
        return ", ".join(parts)

    def _exposure_phrase(self, obs: PersonObservation) -> str:
        flagged: list[str] = []
        for region, value in obs.exposure.items():
            limit = self.exposure_limits.get(region, 1.0)
            if value > limit and region in _REGION_PHRASES:
                flagged.append(f"{_REGION_PHRASES[region]} ({value:.2f} > limit {limit:.2f})")
        if not flagged:
            return ""
        return "; ".join(flagged)

    def explain(
        self,
        obs: PersonObservation,
        scores: Scores,
        decision: Decision,
        matched: list[MatchedRule],
    ) -> str:
        if decision == Decision.INSUFFICIENT_EVIDENCE:
            return (
                f"Evidence quality is low ({obs.evidence_quality:.2f}); not enough reliable "
                "information to assess compliance (likely occlusion, blur, or distance). "
                "No alert raised; queued for optional human review."
            )

        garments = self._garment_phrase(obs)
        exposure = self._exposure_phrase(obs)
        scored = [r for r in matched if not r.advisory_only]
        advisory = [r for r in matched if r.advisory_only]

        if decision == Decision.COMPLIANT and not scored:
            base = f"Detected {garments}. No policy rules were triggered."
            if advisory:
                base += (
                    f" {len(advisory)} advisory signal(s) noted for optional review "
                    f"({', '.join(r.id for r in advisory)})."
                )
            base += f" Compliance score: {scores.compliance_score:.2f}."
            return base

        sentence = f"Detected {garments}"
        if exposure:
            sentence += f" with {exposure}"
        sentence += "."
        if scored:
            n = len(scored)
            sentence += f" {n} rule{'s' if n != 1 else ''} contributed"
            cites = [r.citation for r in scored if r.citation]
            if cites:
                sentence += f" ({'; '.join(cites)})"
            sentence += "."
        sev = _SEVERITY_WORD[decision]
        ov = scores.overall_violation if scores.overall_violation is not None else 0.0
        sentence += f" Campus policy violation score: {ov:.2f} ({sev})."
        sentence += f" Evidence quality: {obs.evidence_quality:.2f}."
        if advisory:
            sentence += (
                f" Advisory (human review only): {', '.join(r.id for r in advisory)}."
            )
        return sentence

    def build_decision(
        self,
        obs: PersonObservation,
        scores: Scores,
        decision: Decision,
        matched: list[MatchedRule],
        remediation: Remediation | None = None,
    ) -> PersonDecision:
        explanation = self.explain(obs, scores, decision, matched)
        coverage = garment_coverage(obs)
        cov = coverage_sentence(coverage)
        if cov:
            explanation = f"{explanation} {cov}"
        cf = remediation_sentence(remediation)
        if cf:
            explanation = f"{explanation} {cf}"
        return PersonDecision(
            track_id=obs.track_id,
            bbox=obs.bbox,
            decision=decision,
            scores=scores,
            observation=obs,
            matched_rules=matched,
            coverage=coverage,
            explanation=explanation,
            remediation=remediation,
            action=choose_action(decision),
        )
