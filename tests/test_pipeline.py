"""End-to-end pipeline + explainability + temporal fusion tests."""

from __future__ import annotations

from clothic.fusion.temporal import TemporalFuser
from clothic.pipeline import ClothicPipeline
from clothic.schemas import Decision, GarmentEvidence, PersonObservation


def test_pipeline_runs_all_bands_with_mock_backend():
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    # Feed several frames so debounce lets persistent violations stick (as in
    # real video). A single frame is intentionally held back to suppress flicker.
    for _ in range(5):
        result = pipe.process_frame(None)
    assert result.profile_id == "default"
    assert len(result.persons) == 4
    decisions = {p.decision for p in result.persons}
    # The default personas are designed to exercise every band.
    assert Decision.COMPLIANT in decisions
    assert Decision.MAJOR_VIOLATION in decisions
    assert Decision.INSUFFICIENT_EVIDENCE in decisions
    for p in result.persons:
        assert p.explanation  # every decision is explained
        assert p.action


def test_explanation_mentions_evidence():
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    major = next(p for p in result.persons if p.decision == Decision.MAJOR_VIOLATION)
    text = major.explanation.lower()
    assert "sleeveless" in text or "shorts" in text
    assert "violation score" in text


def test_single_shot_mode_surfaces_violation_on_first_frame():
    # With temporal disabled (single-image use), a violation must NOT be held
    # back by debounce -- it appears on the very first frame.
    pipe = ClothicPipeline(profile_id="default", backend="mock", enable_temporal=False)
    result = pipe.process_frame(None)
    decisions = {p.decision for p in result.persons}
    assert Decision.MAJOR_VIOLATION in decisions


def test_latency_recorded():
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    result = pipe.process_frame(None)
    assert "total" in result.latency_ms
    assert result.latency_ms["total"] >= 0


def test_temporal_smoothing_blends_attributes():
    fuser = TemporalFuser(alpha=0.5)
    o1 = PersonObservation(
        track_id=7, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tshirt", conf=0.8, attributes={"short_sleeve": 1.0}),
        exposure={"shoulder": 1.0}, evidence_quality=1.0,
    )
    o2 = PersonObservation(
        track_id=7, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tshirt", conf=0.4, attributes={"short_sleeve": 0.0}),
        exposure={"shoulder": 0.0}, evidence_quality=0.0,
    )
    fuser.smooth(o1)
    fused = fuser.smooth(o2)
    # EMA with alpha=0.5 halves the swing rather than jumping to the new frame.
    assert 0.4 < fused.upper.attr("short_sleeve") < 0.6
    assert 0.4 < fused.exposure["shoulder"] < 0.6
    assert fused.frames_seen == 2


def test_debounce_suppresses_single_frame_violation():
    fuser = TemporalFuser(k=3, m=5)
    # A lone violation frame should be held back (reported compliant) while warming up.
    out = fuser.debounce(1, Decision.MAJOR_VIOLATION)
    assert out == Decision.COMPLIANT
    # After K consecutive violation frames it sticks.
    for _ in range(3):
        out = fuser.debounce(1, Decision.MAJOR_VIOLATION)
    assert out == Decision.MAJOR_VIOLATION
