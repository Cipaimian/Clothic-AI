"""Tests for temperature-scaling confidence calibration."""

from __future__ import annotations

import numpy as np

from clothic.reasoning.calibration import (
    Calibrator,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)


def _overconfident_data(n=4000, seed=0):
    """Generate labels whose true prob is a softened version of the score.

    The model reports `probs` but is over-confident: true accuracy follows a
    higher temperature. A good calibrator should recover T > 1.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, size=n)
    reported = 1.0 / (1.0 + np.exp(-logits))            # over-confident scores
    true_p = 1.0 / (1.0 + np.exp(-logits / 2.0))        # actual reliability
    labels = (rng.random(n) < true_p).astype(float)
    return reported, labels


def test_fit_temperature_reduces_ece():
    probs, labels = _overconfident_data()
    before = expected_calibration_error(probs, labels)
    t = fit_temperature(probs, labels)
    calibrated = apply_temperature(probs, t)
    after = expected_calibration_error(calibrated, labels)
    assert t > 1.0                       # over-confidence -> softening
    assert after < before                # calibration improved
    assert after < 0.05                  # and is now small


def test_apply_temperature_scalar_and_array():
    assert abs(apply_temperature(0.5, 2.0) - 0.5) < 1e-9  # 0.5 is a fixed point
    out = apply_temperature(np.array([0.9, 0.1]), 2.0)
    assert out[0] < 0.9 and out[1] > 0.1  # softened toward 0.5


def test_calibrator_per_head():
    probs, labels = _overconfident_data()
    cal = Calibrator()
    t = cal.fit("attr_sleeveless", probs, labels)
    assert cal.temperatures["attr_sleeveless"] == t
    # Unknown head defaults to T=1 (identity).
    assert cal.calibrate("unknown_head", 0.8) == 0.8


def test_ece_perfectly_calibrated_is_low():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    labels = (rng.random(5000) < p).astype(float)  # labels drawn at the true rate
    assert expected_calibration_error(p, labels) < 0.05


def test_calibrate_observation_identity_is_noop():
    from clothic.reasoning.calibration import Calibrator, calibrate_observation
    from clothic.schemas import GarmentEvidence, PersonObservation

    obs = PersonObservation(
        track_id=1, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tank_top", conf=0.9, attributes={"sleeveless": 0.9}),
        exposure={"shoulder": 0.8},
    )
    out = calibrate_observation(obs, Calibrator())  # empty -> identity
    assert out is obs


def test_calibrate_observation_softens_confidences():
    from clothic.reasoning.calibration import Calibrator, calibrate_observation
    from clothic.schemas import GarmentEvidence, PersonObservation

    obs = PersonObservation(
        track_id=1, bbox=(0, 0, 1, 1),
        upper=GarmentEvidence(type="tank_top", conf=0.95, attributes={"sleeveless": 0.95}),
        exposure={"shoulder": 0.9},
    )
    cal = Calibrator({"garment_conf": 2.0, "attr.sleeveless": 2.0, "exposure.shoulder": 2.0})
    out = calibrate_observation(obs, cal)
    assert out.upper.conf < 0.95
    assert out.upper.attr("sleeveless") < 0.95
    assert out.exposure["shoulder"] < 0.9
