"""Tests for the four-score aggregation and decision banding."""

from __future__ import annotations

from clothic.reasoning.rule_engine import RuleEngine
from clothic.reasoning.scoring import ScoringEngine
from clothic.schemas import Decision, GarmentEvidence, PersonObservation


def _obs(quality=0.9, **kw):
    base = dict(track_id=1, bbox=(0, 0, 10, 10), evidence_quality=quality)
    base.update(kw)
    return PersonObservation(**base)


def test_compliant_outfit_scores_high(default_profile):
    engine = RuleEngine(default_profile)
    scorer = ScoringEngine(default_profile)
    obs = _obs(
        upper=GarmentEvidence(type="shirt_formal", conf=0.9, attributes={"long_sleeve": 0.9}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
        footwear=GarmentEvidence(type="formal_shoes", conf=0.85),
        exposure={"shoulder": 0.0, "thigh": 0.0},
    )
    matched = engine.match(obs)
    scores, decision = scorer.score(obs, matched)
    assert decision == Decision.COMPLIANT
    assert scores.compliance_score >= 0.8
    assert scores.overall_violation is not None and scores.overall_violation < 0.2


def test_major_violation_sleeveless_shorts(default_profile):
    engine = RuleEngine(default_profile)
    scorer = ScoringEngine(default_profile)
    obs = _obs(
        upper=GarmentEvidence(type="tank_top", conf=0.91, attributes={"sleeveless": 0.93}),
        lower=GarmentEvidence(type="shorts", conf=0.88, attributes={"hemline_above_knee": 0.82}),
        exposure={"shoulder": 0.78, "thigh": 0.33},
    )
    matched = engine.match(obs)
    scores, decision = scorer.score(obs, matched)
    assert decision == Decision.MAJOR_VIOLATION
    # Two strong rules should push overall violation high (~0.87 region).
    assert scores.overall_violation is not None and scores.overall_violation > 0.8
    assert scores.exposure_score > 0.0


def test_insufficient_evidence_when_occluded(default_profile):
    scorer = ScoringEngine(default_profile)
    obs = _obs(quality=0.3)  # low evidence quality -> high uncertainty
    scores, decision = scorer.score(obs, [])
    assert decision == Decision.INSUFFICIENT_EVIDENCE
    assert scores.overall_violation is None
    assert scores.uncertainty_score > 0.3


def test_advisory_rule_does_not_raise_violation(default_profile):
    engine = RuleEngine(default_profile)
    scorer = ScoringEngine(default_profile)
    obs = _obs(
        upper=GarmentEvidence(type="blouse", conf=0.9, attributes={"transparent_sheer": 0.9}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
    )
    matched = engine.match(obs)
    assert any(r.advisory_only for r in matched)
    scores, decision = scorer.score(obs, matched)
    # Advisory-only signal must not by itself produce a violation.
    assert decision == Decision.COMPLIANT
