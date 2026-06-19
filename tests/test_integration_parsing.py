"""End-to-end integration of the REAL-model path, using a synthetic seg map.

We can't run Sapiens here (no GPU/torch), but we can prove the whole flow is
correct by feeding a hand-painted Goliath-class segmentation map through the
exact production exposure math, then through the real rule engine and scorer:

    synthetic Sapiens seg (Model 1)        garment attributes (Model 2)
              │                                      │
              ▼                                      ▼
    region_exposure_ratio  ───►  PersonObservation  ◄──┘
                                       │
                                       ▼
                          RuleEngine + ScoringEngine
                                       │
                                       ▼
                                   Decision

If this passes, the only untested part of the real path is the neural net
weights themselves - the logic that turns pixels into a compliance verdict is
verified.
"""

from __future__ import annotations

import numpy as np

from clothic.perception.parsing import (
    CLOTHING_CLASSES,
    GOLIATH_CLASSES,
    SKIN_CLASSES,
    class_ids,
    region_exposure_ratio,
    regions_from_pose,
)
from clothic.perception.pose import PoseAnchors
from clothic.reasoning.rule_engine import RuleEngine
from clothic.reasoning.scoring import ScoringEngine
from clothic.schemas import Decision, GarmentEvidence, PersonObservation

LABEL_MAP = dict(enumerate(GOLIATH_CLASSES))
SKIN_IDS = class_ids(LABEL_MAP, SKIN_CLASSES)
CLOTHING_IDS = class_ids(LABEL_MAP, CLOTHING_CLASSES)


def _id(name: str) -> int:
    return next(iter(class_ids(LABEL_MAP, {name})))


def _paint(seg: np.ndarray, window, class_id: int) -> None:
    x1, y1, x2, y2 = window
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(seg.shape[1], x2), min(seg.shape[0], y2)
    seg[y1:y2, x1:x2] = class_id


# A standing person: 100 wide x 200 tall, with realistic anchor lines.
BBOX = (0, 0, 100, 200)
ANCHORS = PoseAnchors(shoulder_y=40, hip_y=110, knee_y=160, ankle_y=200,
                      torso_len=70, visibility=0.9)
REGIONS = regions_from_pose(ANCHORS, BBOX)


def _exposure(scenario: dict[str, str]) -> dict[str, float]:
    """Paint a seg map from {region: 'skin'|'clothing'} and measure exposure."""
    seg = np.zeros((200, 100), dtype=int)  # all Background
    skin_id, cloth_id = _id("Right Upper Arm"), _id("Upper Clothing")
    leg_skin = _id("Right Upper Leg")
    for region, kind in scenario.items():
        cid = cloth_id if kind == "clothing" else (
            leg_skin if region in ("thigh", "knee", "calf") else skin_id)
        _paint(seg, REGIONS[region], cid)
    return region_exposure_ratio(seg, SKIN_IDS, CLOTHING_IDS, REGIONS)


def test_synthetic_tank_top_and_shorts_is_major_violation(default_profile=None):
    from clothic.config import load_profile

    profile = load_profile("default")
    # Model 1: bare shoulders/arms/thighs/knees, torso covered by the tank top.
    exposure = _exposure({
        "shoulder": "skin", "upper_arm": "skin", "midriff": "clothing",
        "thigh": "skin", "knee": "skin", "calf": "skin",
    })
    assert exposure["shoulder"] > 0.9 and exposure["thigh"] > 0.9
    assert exposure["midriff"] < 0.1

    # Model 2: detector says tank top (sleeveless) + shorts (above knee).
    obs = PersonObservation(
        track_id=1, bbox=BBOX,
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.95}),
        lower=GarmentEvidence(type="shorts", conf=0.9, attributes={"hemline_above_knee": 0.9}),
        exposure=exposure, evidence_quality=0.9,
    )
    matched = RuleEngine(profile).match(obs)
    scores, decision = ScoringEngine(profile).score(obs, matched)

    assert {"UPPER_SLEEVELESS", "LOWER_ABOVE_KNEE"} <= {r.id for r in matched}
    assert decision == Decision.MAJOR_VIOLATION
    assert scores.exposure_score > 0.0


def test_synthetic_fully_covered_is_compliant():
    from clothic.config import load_profile

    profile = load_profile("default")
    exposure = _exposure({r: "clothing" for r in REGIONS})  # everything covered
    assert all(v < 0.1 for v in exposure.values())

    obs = PersonObservation(
        track_id=2, bbox=BBOX,
        upper=GarmentEvidence(type="shirt_formal", conf=0.9, attributes={"long_sleeve": 0.9}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
        exposure=exposure, evidence_quality=0.9,
    )
    matched = RuleEngine(profile).match(obs)
    scores, decision = ScoringEngine(profile).score(obs, matched)
    assert decision == Decision.COMPLIANT
    assert scores.exposure_score == 0.0


def test_coverage_complements_synthetic_exposure():
    from clothic.perception.coverage import garment_coverage

    exposure = _exposure({
        "shoulder": "skin", "upper_arm": "skin", "midriff": "clothing",
        "thigh": "clothing", "knee": "clothing", "calf": "clothing",
    })
    obs = PersonObservation(
        track_id=3, bbox=BBOX,
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.95}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
        exposure=exposure,
    )
    reports = {r.garment_type: r for r in garment_coverage(obs)}
    # The tank top barely covers shoulders; the pants fully cover the legs.
    assert reports["tank_top"].regions["shoulder"] < 0.1
    assert reports["long_pants"].regions["thigh"] > 0.9
