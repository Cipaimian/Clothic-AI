"""Counterfactual ("what-if") explanation generator.

Answers the most actionable question a flagged student can ask: *what would
make me compliant?* For each rule that fired, a remediation defines a concrete
intervention on the observation (e.g. "wear a top with sleeves"). The engine is
then re-run on the modified observation to **verify** the suggested changes
actually flip the decision to compliant -- the advice is checked, not asserted.

This is intervention-based: it mutates copies of the ``PersonObservation`` and
re-evaluates, so it stays consistent with whatever policy is loaded. No second
model, nothing learned -- the counterfactual is a property of the same
transparent rule engine that produced the decision.
"""

from __future__ import annotations

from typing import Callable

from clothic.schemas import (
    Decision,
    GarmentEvidence,
    PersonObservation,
    Remediation,
)

# An intervention: a human description + a pure transform on an observation.
Transform = Callable[[PersonObservation], None]


def _cover_arms(obs: PersonObservation) -> None:
    if obs.upper is not None:
        attrs = dict(obs.upper.attributes)
        attrs.update({"sleeveless": 0.0, "short_sleeve": 0.0, "long_sleeve": 1.0})
        obs.upper = GarmentEvidence(type="tshirt", conf=obs.upper.conf, attributes=attrs)
    obs.exposure = {**obs.exposure, "shoulder": 0.0, "upper_arm": 0.0}


def _cover_legs(obs: PersonObservation) -> None:
    if obs.lower is not None:
        attrs = dict(obs.lower.attributes)
        attrs.update(
            {"hemline_above_knee": 0.0, "hemline_at_knee": 0.0, "hemline_below_knee": 1.0}
        )
        obs.lower = GarmentEvidence(type="long_pants", conf=obs.lower.conf, attributes=attrs)
    obs.exposure = {**obs.exposure, "thigh": 0.0, "knee": 0.0, "calf": 0.0}


def _cover_midriff(obs: PersonObservation) -> None:
    if obs.upper is not None:
        attrs = dict(obs.upper.attributes)
        attrs["midriff_exposed"] = 0.0
        obs.upper = GarmentEvidence(type=obs.upper.type, conf=obs.upper.conf, attributes=attrs)
    obs.exposure = {**obs.exposure, "midriff": 0.0}


def _closed_footwear(obs: PersonObservation) -> None:
    conf = obs.footwear.conf if obs.footwear else 0.8
    obs.footwear = GarmentEvidence(type="shoes_closed", conf=conf, attributes={})


def _undamaged(obs: PersonObservation) -> None:
    for slot in ("upper", "lower"):
        g = getattr(obs, slot)
        if g is not None:
            attrs = dict(g.attributes)
            attrs["ripped_torn"] = 0.0
            setattr(obs, slot, GarmentEvidence(type=g.type, conf=g.conf, attributes=attrs))


def _formal_upper(obs: PersonObservation) -> None:
    conf = obs.upper.conf if obs.upper else 0.8
    obs.upper = GarmentEvidence(type="shirt_formal", conf=conf,
                                attributes={"long_sleeve": 1.0, "formal_style": 0.9})
    obs.exposure = {**obs.exposure, "shoulder": 0.0, "upper_arm": 0.0}


# Map a rule category (and a few id hints) to a remediation. Ordered by how
# commonly each applies; the generator only uses the ones whose rules fired.
_REMEDIATIONS: list[tuple[str, str, Transform]] = [
    ("upper_body_sleeve", "wear a top with sleeves", _cover_arms),
    ("upper_body_formal", "wear a collared / formal top", _formal_upper),
    ("upper_body_midriff", "wear a top that covers the midriff", _cover_midriff),
    ("lower_body", "wear bottoms that reach the knee", _cover_legs),
    ("footwear", "wear closed shoes instead of open footwear", _closed_footwear),
    ("condition", "wear undamaged (not torn / ripped) clothing", _undamaged),
]


def _category_key(rule_id: str, category: str) -> str:
    """Disambiguate upper-body rules so the right remediation is chosen."""
    rid = rule_id.upper()
    if category == "upper_body":
        if "SLEEVELESS" in rid:
            return "upper_body_sleeve"
        if "MIDRIFF" in rid or "CROP" in rid:
            return "upper_body_midriff"
        if "FORMAL" in rid:
            return "upper_body_formal"
        return "upper_body_sleeve"
    return category


class CounterfactualEngine:
    def __init__(self, rule_engine, scoring):
        self.rule_engine = rule_engine
        self.scoring = scoring

    def generate(
        self,
        obs: PersonObservation,
        matched_rules,
        zone: str | None = None,
    ) -> Remediation | None:
        """Return a verified minimal remediation, or None if not applicable."""
        scored = [r for r in matched_rules if not r.advisory_only]
        if not scored:
            return None

        # Pick the relevant remediations for the rules that actually fired.
        wanted_keys = {_category_key(r.id, r.category) for r in scored}
        chosen = [(desc, fn) for key, desc, fn in _REMEDIATIONS if key in wanted_keys]
        if not chosen:
            return None

        # Apply all chosen interventions to a deep copy and verify compliance.
        trial = obs.model_copy(deep=True)
        for _desc, fn in chosen:
            fn(trial)
        new_matched = self.rule_engine.match(trial, zone=zone)
        _scores, new_decision = self.scoring.score(trial, new_matched)

        return Remediation(
            steps=[desc for desc, _ in chosen],
            addresses_rules=[r.id for r in scored],
            resulting_decision=new_decision,
            verified=(new_decision == Decision.COMPLIANT),
        )


def remediation_sentence(rem: Remediation | None) -> str:
    """Render a remediation as one human sentence for explanations/UI."""
    if rem is None or not rem.steps:
        return ""
    joined = "; ".join(rem.steps)
    if rem.verified:
        return f"To become compliant: {joined}."
    return f"Suggested changes (may not fully resolve all rules): {joined}."
