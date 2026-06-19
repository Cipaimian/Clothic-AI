"""Tests for per-garment coverage attribution (Model 1 ⟶ Model 2 linkage)."""

from __future__ import annotations

from clothic.perception.coverage import coverage_sentence, garment_coverage
from clothic.pipeline import ClothicPipeline
from clothic.schemas import Decision, GarmentEvidence, PersonObservation


def test_coverage_is_one_minus_exposure():
    obs = PersonObservation(
        track_id=1, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tshirt", conf=0.9, attributes={"short_sleeve": 0.9}),
        lower=GarmentEvidence(type="long_pants", conf=0.9, attributes={"hemline_below_knee": 0.95}),
        exposure={"shoulder": 0.0, "upper_arm": 0.05, "midriff": 0.0,
                  "thigh": 0.0, "knee": 0.0, "calf": 0.0},
    )
    reports = garment_coverage(obs)
    by_type = {r.garment_type: r for r in reports}
    assert by_type["tshirt"].regions["shoulder"] == 1.0
    assert by_type["tshirt"].regions["upper_arm"] == 0.95
    assert by_type["long_pants"].regions["thigh"] == 1.0
    assert by_type["long_pants"].slot == "lower"


def test_exposed_garment_low_coverage():
    obs = PersonObservation(
        track_id=2, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.95}),
        exposure={"shoulder": 0.85, "upper_arm": 0.75, "midriff": 0.0},
    )
    reports = garment_coverage(obs)
    tank = reports[0]
    assert tank.regions["shoulder"] < 0.2   # barely covers shoulders
    assert "tank_top covers" in coverage_sentence(reports)


def test_no_garment_no_report():
    obs = PersonObservation(track_id=3, bbox=(0, 0, 1, 1), exposure={"shoulder": 0.5})
    assert garment_coverage(obs) == []
    assert coverage_sentence([]) == ""


def test_pipeline_populates_coverage_field():
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    compliant = next(p for p in result.persons if p.decision == Decision.COMPLIANT)
    assert compliant.coverage  # structured coverage present
    assert "covers" in compliant.explanation.lower()
