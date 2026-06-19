"""Tests for verified counterfactual ('what-if') explanations."""

from __future__ import annotations

from clothic.explain.counterfactual import CounterfactualEngine, remediation_sentence
from clothic.pipeline import ClothicPipeline
from clothic.reasoning.rule_engine import RuleEngine
from clothic.reasoning.scoring import ScoringEngine
from clothic.schemas import Decision, GarmentEvidence, PersonObservation


def _engine(profile):
    return CounterfactualEngine(RuleEngine(profile), ScoringEngine(profile))


def test_counterfactual_flips_sleeveless_shorts_to_compliant(default_profile):
    engine = RuleEngine(default_profile)
    cf = _engine(default_profile)
    obs = PersonObservation(
        track_id=1, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.93}),
        lower=GarmentEvidence(type="shorts", conf=0.9, attributes={"hemline_above_knee": 0.82}),
        exposure={"shoulder": 0.78, "thigh": 0.33},
    )
    matched = engine.match(obs)
    rem = cf.generate(obs, matched)
    assert rem is not None
    assert rem.verified is True
    assert rem.resulting_decision == Decision.COMPLIANT
    # It should suggest covering arms AND legs (both rules fired).
    joined = " ".join(rem.steps).lower()
    assert "sleeve" in joined
    assert "knee" in joined
    assert {"UPPER_SLEEVELESS", "LOWER_ABOVE_KNEE"} <= set(rem.addresses_rules)


def test_counterfactual_footwear_only(default_profile):
    engine = RuleEngine(default_profile)
    cf = _engine(default_profile)
    obs = PersonObservation(
        track_id=2, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tshirt", conf=0.9, attributes={"short_sleeve": 0.9}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
        footwear=GarmentEvidence(type="sandals", conf=0.8),
    )
    rem = cf.generate(obs, engine.match(obs))
    assert rem is not None and rem.verified
    assert "closed shoes" in " ".join(rem.steps).lower()


def test_no_counterfactual_when_compliant(default_profile):
    cf = _engine(default_profile)
    obs = PersonObservation(track_id=3, bbox=(0, 0, 1, 1))
    assert cf.generate(obs, []) is None


def test_remediation_sentence_phrasing():
    from clothic.schemas import Remediation

    rem = Remediation(steps=["wear a top with sleeves"], verified=True,
                      resulting_decision=Decision.COMPLIANT)
    assert remediation_sentence(rem).startswith("To become compliant:")
    assert remediation_sentence(None) == ""


def test_pipeline_attaches_remediation_to_violations():
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    major = next(p for p in result.persons if p.decision == Decision.MAJOR_VIOLATION)
    assert major.remediation is not None
    assert "to become compliant" in major.explanation.lower()
