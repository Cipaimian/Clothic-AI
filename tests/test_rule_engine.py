"""Tests for the transparent rule engine."""

from __future__ import annotations

from clothic.reasoning.rule_engine import RuleEngine, build_attribute_vector, evaluate_predicate
from clothic.schemas import GarmentEvidence, PersonObservation


def _obs(**kw) -> PersonObservation:
    base = dict(track_id=1, bbox=(0, 0, 10, 10))
    base.update(kw)
    return PersonObservation(**base)


def test_attribute_vector_flattening():
    obs = _obs(
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.93}),
        exposure={"shoulder": 0.78},
    )
    vec = build_attribute_vector(obs)
    assert vec["upper.type"] == "tank_top"
    assert vec["upper.sleeveless"] == 0.93
    assert vec["exposure.shoulder"] == 0.78
    # Absent numeric attribute resolves to 0.0, not an error.
    assert vec.get("lower.type") is None


def test_leaf_operators():
    vec = {"upper.sleeveless": 0.8, "upper.type": "tank_top", "exposure.thigh": 0.1}
    assert evaluate_predicate(vec, {"attr": "upper.sleeveless", "op": ">=", "value": 0.6})
    assert not evaluate_predicate(vec, {"attr": "exposure.thigh", "op": ">", "value": 0.2})
    assert evaluate_predicate(vec, {"attr": "upper.type", "op": "in", "value": ["tank_top"]})
    assert evaluate_predicate(vec, {"attr": "upper.type", "op": "not_in", "value": ["tshirt"]})


def test_boolean_composition():
    vec = {"a": 0.9, "b": 0.1}
    pred = {"all": [
        {"attr": "a", "op": ">", "value": 0.5},
        {"any": [
            {"attr": "b", "op": ">", "value": 0.5},
            {"not": {"attr": "b", "op": ">", "value": 0.5}},
        ]},
    ]}
    assert evaluate_predicate(vec, pred)


def test_missing_attribute_does_not_fire():
    # A rule on lower clothing must not fire when there is no lower garment.
    vec = {"upper.type": "tshirt"}
    assert not evaluate_predicate(vec, {"attr": "lower.hemline_above_knee", "op": ">=", "value": 0.6})


def test_engine_matches_sleeveless_and_shorts(default_profile):
    engine = RuleEngine(default_profile)
    obs = _obs(
        upper=GarmentEvidence(type="tank_top", conf=0.91, attributes={"sleeveless": 0.93}),
        lower=GarmentEvidence(type="shorts", conf=0.88, attributes={"hemline_above_knee": 0.82}),
        exposure={"thigh": 0.33, "shoulder": 0.78},
    )
    matched_ids = {r.id for r in engine.match(obs)}
    assert "UPPER_SLEEVELESS" in matched_ids
    assert "LOWER_ABOVE_KNEE" in matched_ids


def test_zone_scoping(default_profile):
    # default profile has no zone-scoped rules; lab profile does.
    from clothic.config import load_profile

    lab = load_profile("lab_safety")
    engine = RuleEngine(lab)
    obs = _obs(footwear=GarmentEvidence(type="sandals", conf=0.8))
    # Outside the lab zone, the lab-only footwear rule must not fire.
    assert not any(r.id == "LAB_CLOSED_FOOTWEAR" for r in engine.match(obs, zone="library"))
    assert any(r.id == "LAB_CLOSED_FOOTWEAR" for r in engine.match(obs, zone="lab"))
