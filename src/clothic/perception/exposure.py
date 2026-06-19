"""Geometric exposure estimation.

Exposure is *computed*, never predicted by a single black-box model, so the
result is auditable and defensible in an appeal. In a real backend the inputs
come from human parsing (skin vs. garment pixels per region) and pose (the
knee/shoulder landmarks that define coverage lines). This module holds the
pure geometry so it can be unit-tested independently of any neural network.
"""

from __future__ import annotations

from clothic.schemas import GarmentEvidence

# Which garment attributes imply coverage / exposure of which body regions.
# This mapping is physical fact, not policy: a sleeveless top exposes shoulders.
REGIONS = ["shoulder", "upper_arm", "midriff", "thigh", "knee", "calf"]


def exposure_from_attributes(
    upper: GarmentEvidence | None,
    lower: GarmentEvidence | None,
) -> dict[str, float]:
    """Derive a per-region exposure ratio from garment attributes.

    This is the lightweight estimator used when full parsing masks are not
    available (e.g. the mock/CPU path). A parsing-based backend should override
    these with measured skin-pixel ratios; the schema and downstream code are
    identical either way.
    """
    exp: dict[str, float] = {r: 0.0 for r in REGIONS}

    if upper is not None:
        sleeveless = upper.attr("sleeveless")
        short_sleeve = upper.attr("short_sleeve")
        midriff = upper.attr("midriff_exposed")
        # Sleeveless exposes shoulders + most of the upper arm.
        exp["shoulder"] = max(exp["shoulder"], 0.85 * sleeveless)
        exp["upper_arm"] = max(exp["upper_arm"], 0.75 * sleeveless + 0.5 * short_sleeve)
        exp["midriff"] = max(exp["midriff"], midriff)

    if lower is not None:
        above_knee = lower.attr("hemline_above_knee")
        at_knee = lower.attr("hemline_at_knee")
        # Above-knee hemlines expose thigh and knee; calves depend on length.
        exp["thigh"] = max(exp["thigh"], 0.9 * above_knee)
        exp["knee"] = max(exp["knee"], above_knee + 0.5 * at_knee)
        exp["calf"] = max(exp["calf"], 0.6 * above_knee)

    return {k: float(min(1.0, v)) for k, v in exp.items()}


def evidence_quality(
    detection_conf: float,
    occlusion_ratio: float = 0.0,
    crop_resolution: float = 1.0,
    pose_visibility: float = 1.0,
    motion_blur: float = 0.0,
) -> float:
    """Combine reliability signals into a single 0..1 quality factor.

    Low quality (heavy occlusion, tiny/blurry crop, hidden landmarks) lowers
    fused confidence so the rule engine routes to ``insufficient_evidence``
    instead of risking a false accusation.
    """
    q = detection_conf
    q *= (1.0 - min(1.0, max(0.0, occlusion_ratio)))
    q *= min(1.0, max(0.0, crop_resolution))
    q *= min(1.0, max(0.0, pose_visibility))
    q *= (1.0 - 0.5 * min(1.0, max(0.0, motion_blur)))
    return float(min(1.0, max(0.0, q)))
