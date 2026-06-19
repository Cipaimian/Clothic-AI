"""Tests for the Sapiens-based exposure math (Model 1), without any model.

The neural net is not needed to validate the project-critical logic:
exposure = skin_pixels / (skin_pixels + clothing_pixels) per region.
"""

from __future__ import annotations

import numpy as np

from clothic.perception.parsing import (
    GOLIATH_CLASSES,
    SKIN_CLASSES,
    class_ids,
    coverage_by_region,
    region_exposure_ratio,
)


def test_goliath_vocab_has_28_classes_and_skin_clothing_split():
    assert len(GOLIATH_CLASSES) == 28
    label_map = dict(enumerate(GOLIATH_CLASSES))
    skin = class_ids(label_map, SKIN_CLASSES)
    cloth = class_ids(label_map, {"Upper Clothing", "Lower Clothing"})
    assert skin and cloth
    assert skin.isdisjoint(cloth)  # bare anatomy and clothing never overlap
    # Torso (bare midriff) is skin; Upper Clothing is clothing.
    assert label_map[next(iter(class_ids(label_map, {"Torso"})))] == "Torso"


def test_exposure_ratio_half_covered():
    # Left half = skin (id 1), right half = clothing (id 2).
    seg = np.zeros((10, 10), dtype=int)
    seg[:, :5] = 1
    seg[:, 5:] = 2
    skin_ids, clothing_ids = {1}, {2}
    regions = {
        "all":   (0, 0, 10, 10),
        "left":  (0, 0, 5, 10),
        "right": (5, 0, 10, 10),
    }
    exp = region_exposure_ratio(seg, skin_ids, clothing_ids, regions)
    assert exp["all"] == 0.5      # half skin, half clothing
    assert exp["left"] == 1.0     # fully exposed
    assert exp["right"] == 0.0    # fully covered


def test_coverage_is_complement_of_exposure():
    exp = {"thigh": 0.9, "shoulder": 0.0}
    cov = coverage_by_region(exp)
    assert cov["thigh"] == pytest_approx(0.1)
    assert cov["shoulder"] == 1.0


def test_background_only_region_is_not_exposed():
    # A region with neither skin nor clothing pixels must read 0.0, not crash.
    seg = np.zeros((10, 10), dtype=int)  # all background (id 0)
    exp = region_exposure_ratio(seg, {1}, {2}, {"r": (0, 0, 10, 10)})
    assert exp["r"] == 0.0


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()
