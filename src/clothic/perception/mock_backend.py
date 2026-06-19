"""Scenario-driven mock perception backend.

Produces deterministic ``PersonObservation`` objects with no heavy dependencies
so the reasoning core, API, CLI, and tests run anywhere. Each "persona" is a
realistic attribute vector; the default scenario rotates through cases that
exercise every decision band (compliant, minor, major, insufficient).
"""

from __future__ import annotations

from typing import Any

from clothic.perception.base import PerceptionBackend
from clothic.perception.exposure import evidence_quality, exposure_from_attributes
from clothic.schemas import GarmentEvidence, PersonObservation

# Built-in personas. Real backends compute these from pixels; here they are
# scripted so behaviour is fully reproducible.
DEFAULT_PERSONAS: list[dict[str, Any]] = [
    {
        "name": "compliant_formal",
        "bbox": [120, 80, 200, 520],
        "upper": {"type": "shirt_formal", "conf": 0.92,
                  "attributes": {"long_sleeve": 0.9, "formal_style": 0.8}},
        "lower": {"type": "long_pants", "conf": 0.90,
                  "attributes": {"hemline_below_knee": 0.95, "formal_style": 0.6}},
        "footwear": {"type": "formal_shoes", "conf": 0.85, "attributes": {}},
        "detection_conf": 0.93, "occlusion": 0.02, "resolution": 1.0,
    },
    {
        "name": "major_sleeveless_shorts",
        "bbox": [412, 96, 233, 564],
        "upper": {"type": "tank_top", "conf": 0.91,
                  "attributes": {"sleeveless": 0.93, "transparent_sheer": 0.08}},
        "lower": {"type": "shorts", "conf": 0.88,
                  "attributes": {"hemline_above_knee": 0.82}},
        "footwear": {"type": "sneakers", "conf": 0.79, "attributes": {}},
        "detection_conf": 0.90, "occlusion": 0.05, "resolution": 0.95,
    },
    {
        "name": "minor_sandals_only",
        "bbox": [640, 110, 210, 500],
        "upper": {"type": "tshirt", "conf": 0.88,
                  "attributes": {"short_sleeve": 0.9}},
        "lower": {"type": "long_pants", "conf": 0.86,
                  "attributes": {"hemline_below_knee": 0.92}},
        "footwear": {"type": "sandals", "conf": 0.80, "attributes": {}},
        "detection_conf": 0.87, "occlusion": 0.05, "resolution": 0.95,
    },
    {
        "name": "insufficient_occluded",
        "bbox": [880, 200, 150, 300],
        "upper": {"type": "hoodie", "conf": 0.45, "attributes": {"long_sleeve": 0.5}},
        "lower": None,
        "footwear": None,
        "detection_conf": 0.50, "occlusion": 0.7, "resolution": 0.4,
    },
]


def _garment(spec: dict[str, Any] | None) -> GarmentEvidence | None:
    if spec is None:
        return None
    return GarmentEvidence(
        type=spec["type"], conf=spec.get("conf", 0.0), attributes=spec.get("attributes", {})
    )


class MockBackend(PerceptionBackend):
    name = "mock"

    def __init__(
        self,
        personas: list[dict[str, Any]] | None = None,
        persons_per_frame: int | None = None,
    ):
        self.personas = personas or DEFAULT_PERSONAS
        # How many personas to surface per frame (default: all of them).
        self.persons_per_frame = persons_per_frame or len(self.personas)
        self._seen: dict[int, int] = {}

    def observe(self, frame: Any = None, frame_index: int = 0) -> list[PersonObservation]:
        out: list[PersonObservation] = []
        for track_id in range(self.persons_per_frame):
            persona = self.personas[track_id % len(self.personas)]
            upper = _garment(persona.get("upper"))
            lower = _garment(persona.get("lower"))
            footwear = _garment(persona.get("footwear"))
            exposure = exposure_from_attributes(upper, lower)
            quality = evidence_quality(
                detection_conf=persona.get("detection_conf", 0.9),
                occlusion_ratio=persona.get("occlusion", 0.0),
                crop_resolution=persona.get("resolution", 1.0),
            )
            self._seen[track_id] = self._seen.get(track_id, 0) + 1
            out.append(
                PersonObservation(
                    track_id=track_id,
                    bbox=tuple(persona["bbox"]),
                    upper=upper,
                    lower=lower,
                    footwear=footwear,
                    exposure=exposure,
                    evidence_quality=quality,
                    frames_seen=self._seen[track_id],
                )
            )
        return out
