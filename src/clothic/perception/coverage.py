"""Per-garment coverage attribution.

Ties Model 2 (which garments) to Model 1 (which body regions are covered). For
each detected garment we report the body regions it is responsible for and how
well it covers them: ``coverage(region) = 1 - exposure(region)``.

This makes the system answer the question directly: *does each garment cover the
body part scanned by the parser?* It is a pure function of the ``exposure`` dict,
so it works with any backend (mock or real) and needs no model here.
"""

from __future__ import annotations

from clothic.schemas import GarmentCoverage, PersonObservation

# Which body regions each clothing slot is expected to cover.
REGION_BY_SLOT: dict[str, list[str]] = {
    "upper": ["shoulder", "upper_arm", "midriff"],
    "lower": ["thigh", "knee", "calf"],
}


def garment_coverage(obs: PersonObservation) -> list[GarmentCoverage]:
    """Return per-garment coverage of its responsible body regions."""
    reports: list[GarmentCoverage] = []
    for slot, regions in REGION_BY_SLOT.items():
        garment = getattr(obs, slot)
        if garment is None:
            continue
        cov = {
            r: round(1.0 - float(obs.exposure.get(r, 0.0)), 3)
            for r in regions
            if r in obs.exposure
        }
        if cov:
            reports.append(
                GarmentCoverage(slot=slot, garment_type=garment.type, regions=cov)
            )
    return reports


def coverage_sentence(reports: list[GarmentCoverage]) -> str:
    """One compact human line summarising what covers what."""
    if not reports:
        return ""
    parts = []
    for r in reports:
        regions = ", ".join(f"{name} {val:.2f}" for name, val in r.regions.items())
        parts.append(f"{r.garment_type} covers {regions}")
    return "Coverage - " + "; ".join(parts) + "."
