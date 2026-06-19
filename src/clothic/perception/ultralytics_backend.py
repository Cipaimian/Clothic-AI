"""Real perception backend built on Ultralytics YOLO + ByteTrack.

This is the production path. It is intentionally import-guarded: the package
imports fine without ``ultralytics``/``torch`` installed, and this module only
loads them when actually instantiated. Install with::

    pip install "clothic[perception]"

Pipeline implemented here:
  * person detection + tracking (YOLO + ByteTrack, ``persist=True``)
  * optional garment detection model (e.g. fine-tuned on your campus dataset)
  * garment-class -> canonical-type/attribute mapping via the ontology
  * geometric exposure estimation (attribute-based; upgrade to parsing later)

NOTE: Human parsing (SegFormer) and pose (RTMPose) are the documented next
upgrade. Their hooks are marked TODO; until then exposure uses the attribute
estimator, and ``evidence_quality`` reflects the reduced certainty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clothic.perception.base import PerceptionBackend
from clothic.perception.exposure import evidence_quality, exposure_from_attributes
from clothic.schemas import GarmentEvidence, PersonObservation

# Maps raw dataset class names -> (slot, canonical_type, {attribute: value}).
# Covers the user's Roboflow campus classes plus common DeepFashion2 names.
DEFAULT_CLASS_MAP: dict[str, tuple[str, str, dict[str, float]]] = {
    # Student Dress Code dataset (Roboflow v3, 13 classes) -- current default.
    "sleeveless": ("upper", "sleeveless_top", {"sleeveless": 0.95}),
    "longsleeved": ("upper", "shirt_long", {"long_sleeve": 0.9}),
    "shirt": ("upper", "shirt", {"short_sleeve": 0.7}),
    "cropped_top": ("upper", "crop_top", {"crop": 0.9, "sleeveless": 0.5}),
    "pants": ("lower", "long_pants", {"hemline_below_knee": 0.95}),
    "ripped_jeans": ("lower", "long_pants", {"ripped_torn": 0.95, "hemline_below_knee": 0.9}),
    "mini-skirt": ("lower", "skirt_short", {"hemline_above_knee": 0.95}),
    "skirt": ("lower", "skirt", {}),       # length ambiguous -> pixel exposure decides
    "dress": ("lower", "dress", {}),       # length ambiguous -> pixel exposure decides
    "shoe": ("footwear", "shoes_closed", {}),
    "hat": ("headwear", "hat", {}),
    # 'id' (ID card) intentionally absent -- not apparel.
    # Legacy campus Roboflow set (Indonesian labels) -- kept for the old best.pt.
    "kaos": ("upper", "tshirt", {"short_sleeve": 0.9}),
    "singlet": ("upper", "tank_top", {"sleeveless": 0.95}),
    "celana_panjang": ("lower", "long_pants", {"hemline_below_knee": 0.95}),
    "celana_pendek": ("lower", "shorts", {"hemline_above_knee": 0.9}),
    # A few DeepFashion2-style names for transfer
    "short_sleeve_top": ("upper", "tshirt", {"short_sleeve": 0.9}),
    "long_sleeve_top": ("upper", "shirt_formal", {"long_sleeve": 0.9}),
    "vest": ("upper", "sleeveless_top", {"sleeveless": 0.9}),
    "shorts": ("lower", "shorts", {"hemline_above_knee": 0.9}),
    "trousers": ("lower", "long_pants", {"hemline_below_knee": 0.95}),
}


class UltralyticsBackend(PerceptionBackend):
    name = "ultralytics"

    def __init__(
        self,
        person_weights: str = "yolo11n.pt",
        garment_weights: str | None = None,
        class_map: dict[str, tuple[str, str, dict[str, float]]] | None = None,
        device: str | None = None,
        conf: float = 0.35,
    ):
        from ultralytics import YOLO  # imported lazily so base package stays light

        self.person_model = YOLO(person_weights)
        self.garment_model = YOLO(garment_weights) if garment_weights else None
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.device = device
        self.conf = conf

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _iou_contains(person_box, gbox) -> float:
        """Fraction of garment box area that lies inside the person box."""
        px1, py1, px2, py2 = person_box
        gx1, gy1, gx2, gy2 = gbox
        ix1, iy1 = max(px1, gx1), max(py1, gy1)
        ix2, iy2 = min(px2, gx2), min(py2, gy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        garea = max(1e-6, (gx2 - gx1) * (gy2 - gy1))
        return inter / garea

    def _detect_garments(self, frame) -> list[tuple[tuple, str, float]]:
        if self.garment_model is None:
            return []
        res = self.garment_model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        names = res.names
        out = []
        for box in res.boxes:
            cls_name = names[int(box.cls)]
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            out.append((xyxy, cls_name, float(box.conf)))
        return out

    # -- main --------------------------------------------------------------

    def observe(self, frame: Any, frame_index: int = 0) -> list[PersonObservation]:
        res = self.person_model.track(
            frame, persist=True, classes=[0], conf=self.conf,
            device=self.device, tracker="bytetrack.yaml", verbose=False,
        )[0]

        garments = self._detect_garments(frame)
        observations: list[PersonObservation] = []

        if res.boxes is None:
            return observations

        for box in res.boxes:
            track_id = int(box.id) if box.id is not None else -1
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            person_box = (x1, y1, x2, y2)
            pconf = float(box.conf)

            slots: dict[str, GarmentEvidence] = {}
            for gbox, cls_name, gconf in garments:
                if self._iou_contains(person_box, gbox) < 0.5:
                    continue
                mapping = self.class_map.get(cls_name)
                if mapping is None:
                    continue
                slot, ctype, attrs = mapping
                # Keep the highest-confidence garment per slot.
                if slot not in slots or gconf > slots[slot].conf:
                    slots[slot] = GarmentEvidence(type=ctype, conf=gconf, attributes=dict(attrs))

            upper = slots.get("upper")
            lower = slots.get("lower")
            footwear = slots.get("footwear")
            exposure = exposure_from_attributes(upper, lower)
            # Evidence quality reflects how well we both *located* the person and
            # *recognised* the clothing (the thing actually being judged). A
            # confident garment detection should not be gated out as "uncertain".
            garment_confs = [g.conf for g in (upper, lower, footwear) if g is not None]
            # What is actually judged is the clothing, so weight its recognition
            # confidence above the person-box localisation confidence. A person
            # box with no recognised garments stays low-quality (-> abstain).
            if garment_confs:
                clothing_conf = sum(garment_confs) / len(garment_confs)
                combined_conf = 0.3 * pconf + 0.7 * clothing_conf
            else:
                combined_conf = 0.5 * pconf
            # Mild discount: exposure is attribute-derived here, not pixel-measured
            # (the FullBackend with Sapiens removes this discount entirely).
            quality = 0.95 * evidence_quality(detection_conf=combined_conf)

            observations.append(
                PersonObservation(
                    track_id=track_id,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    upper=upper,
                    lower=lower,
                    footwear=footwear,
                    exposure=exposure,
                    evidence_quality=quality,
                    frames_seen=1,
                )
            )
        return observations
